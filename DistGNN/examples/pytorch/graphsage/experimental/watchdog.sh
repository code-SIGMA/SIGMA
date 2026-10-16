#!/usr/bin/env bash
#
# watchdog.sh - Wrapper: runs a command, writes its output to a file via
# tee, and terminates it if no new output is received for N consecutive
# check intervals. As soon as output resumes, the counter is reset.
#
# Termination is escalated in stages and verified:
#   1. SIGTERM to the process group
#   2. SIGTERM to every process left in the command's SESSION (catches
#      processes that changed their process group)
#   3. SIGKILL to group + session
#   4. optional: pkill -f PATTERN locally  (-p)
#   5. optional: pkill -f PATTERN on all hosts of a hostfile via ssh (-H)
#   6. optional: arbitrary cleanup command (-x)
#   7. report anything that is still alive
#
# A guard process (dead man's switch) cleans up even if the watchdog itself
# is SIGKILLed. Disable with -G.
#
# Usage:  ./watchdog.sh [OPTIONS] COMMAND [ARGS...]
set -uo pipefail

INTERVAL="${INTERVAL:-120}"          # seconds between two checks
MAX_IDLE="${MAX_IDLE:-25}"           # this many consecutive empty checks -> abort
LOGFILE="${LOGFILE:-output.log}"     # tee writes here
GRACE="${GRACE:-5}"                  # seconds between SIGTERM and SIGKILL
PATTERN="${PATTERN:-}"               # pkill -f pattern for stragglers
HOSTFILE="${HOSTFILE:-}"             # hosts for the remote sweep
EXTRA_CLEAN="${EXTRA_CLEAN:-}"       # arbitrary cleanup command
GUARD=1                              # dead man's switch active
TIMEOUT_RC=124                       # exit code when terminated by the watchdog

usage() {
    cat <<EOF
Usage: $0 [-i SEC] [-n COUNT] [-f FILE] [-g SEC] [-a] [-p PAT] [-H FILE] [-x CMD] [-G] COMMAND [ARGS...]
  -i SEC     check interval in seconds              (default: $INTERVAL)
  -n COUNT   empty checks before termination        (default: $MAX_IDLE)
  -f FILE    log file for tee                       (default: $LOGFILE)
  -g SEC     grace period from TERM -> KILL         (default: $GRACE)
  -a         append to the log file instead of truncating it
  -p PAT     additionally 'pkill -f PAT' for processes that escaped the session
  -H FILE    hostfile: run the -p sweep on every host via ssh (MPI ranks!)
  -x CMD     run this command at the very end of the cleanup
  -G         disable the guard process (dead man's switch)
EOF
}

APPEND=0
while getopts ":i:n:f:g:p:H:x:aGh" opt; do
    case "$opt" in
        i) INTERVAL=$OPTARG ;;
        n) MAX_IDLE=$OPTARG ;;
        f) LOGFILE=$OPTARG ;;
        g) GRACE=$OPTARG ;;
        p) PATTERN=$OPTARG ;;
        H) HOSTFILE=$OPTARG ;;
        x) EXTRA_CLEAN=$OPTARG ;;
        a) APPEND=1 ;;
        G) GUARD=0 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[ "$#" -gt 0 ] || { usage >&2; exit 2; }
[ "$APPEND" -eq 1 ] || : > "$LOGFILE"

log()  { printf '[watchdog] %s\n' "$*" >&2; }
size() { wc -c < "$LOGFILE" 2>/dev/null || echo 0; }   # portable (instead of stat -c %s)

cmd_pid=""
cmd_pgid=""
cmd_sid=""
sleep_pid=""
guard_pid=""

# --------------------------------------------------------------- /proc helpers
# /proc/PID/stat after the ")" of comm:  1=state 2=ppid 3=pgrp 4=session
proc_field() {                      # $1=pid  $2=index (1..4)
    local rest idx=$2
    rest=$(sed 's/.*) //' "/proc/$1/stat" 2>/dev/null) || return 1
    set -- $rest
    eval "printf '%s\n' \"\${$idx:-}\""
}

# All live PIDs belonging to the command's session. Zombies are skipped -
# they still occupy a /proc entry but are already dead.
session_pids() {
    [ -n "$cmd_sid" ] || return 0
    local d p rest
    for d in /proc/[0-9]*; do
        p=${d#/proc/}
        [ "$p" = "$$" ] && continue
        rest=$(sed 's/.*) //' "$d/stat" 2>/dev/null) || continue
        set -- $rest
        [ "${1:-}" = "Z" ] && continue
        [ "${4:-}" = "$cmd_sid" ] && printf '%s\n' "$p"
    done
}

count_session() { session_pids | wc -l | tr -d ' '; }

signal_session() {                  # $1=signal -> prints number of signalled PIDs
    local p n=0
    for p in $(session_pids); do
        kill -"$1" "$p" 2>/dev/null && n=$((n + 1))
    done
    printf '%s\n' "$n"
}

# Signal the whole process group (covers what /proc scanning might miss).
stop_group() {
    [ -n "$cmd_pgid" ] && kill -"$1" -- -"$cmd_pgid" 2>/dev/null
    [ -n "$cmd_pid"  ] && kill -"$1" "$cmd_pid"      2>/dev/null
    return 0
}

wait_gone() {                       # $1=seconds
    local deadline=$((SECONDS + $1))
    while [ "$SECONDS" -lt "$deadline" ]; do
        [ "$(count_session)" -eq 0 ] && return 0
        sleep 0.2
    done
    return 1
}

# --------------------------------------------------------------- extra sweeps
pattern_sweep() {
    [ -n "$PATTERN" ] || return 0
    command -v pgrep >/dev/null 2>&1 || { log "pgrep missing - skipping -p sweep"; return 0; }
    local pids sig
    for sig in TERM KILL; do
        # Exclude self and parent: the watchdog's own argv contains the pattern.
        pids=$(pgrep -f -- "$PATTERN" 2>/dev/null | grep -vx -e "$$" -e "${PPID:-0}" | tr '\n' ' ')
        [ -n "${pids// /}" ] || return 0
        log "local straggler sweep ($sig): $pids"
        kill -"$sig" $pids 2>/dev/null
        [ "$sig" = TERM ] && sleep 2
    done
}

remote_sweep() {
    [ -n "$PATTERN" ] && [ -n "$HOSTFILE" ] && [ -r "$HOSTFILE" ] || return 0
    command -v ssh >/dev/null 2>&1 || { log "ssh missing - skipping remote sweep"; return 0; }
    local line h pids="" w
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        h=${line%%:*}; h=${h%% *}
        [ -n "$h" ] || continue
        log "remote cleanup on $h"
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$h" \
            "pkill -TERM -f -- '$PATTERN'; sleep 3; pkill -KILL -f -- '$PATTERN'" \
            </dev/null >/dev/null 2>&1 &
        pids="$pids $!"
    done < "$HOSTFILE"
    for w in $pids; do wait "$w" 2>/dev/null; done
}

report_survivors() {
    local p left
    left=$(session_pids)
    [ -n "$left" ] || return 0
    log "WARNING: processes still alive after cleanup:"
    for p in $left; do
        log "  PID $p: $(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-110)"
    done
}

# --------------------------------------------------------------- cleanup
# Runs on EVERY exit path (normal end, Ctrl+C, TERM, HUP, error, ...).
cleanup() {
    trap '' INT TERM HUP EXIT                      # no re-entry while cleaning up
    [ -n "$sleep_pid" ] && kill "$sleep_pid" 2>/dev/null

    if [ -n "$cmd_pid" ]; then
        if kill -0 "$cmd_pid" 2>/dev/null || [ "$(count_session)" -gt 0 ]; then
            log "stopping command (PID $cmd_pid, PGID $cmd_pgid, SID $cmd_sid) ..."
            stop_group TERM
            if ! wait_gone "$GRACE"; then
                n=$(signal_session TERM)
                if [ "$n" -gt 0 ]; then
                    log "$n process(es) outside the group - SIGTERM sent directly"
                    wait_gone 3
                fi
            fi
            if [ "$(count_session)" -gt 0 ]; then
                log "$(count_session) process(es) ignoring SIGTERM - sending SIGKILL"
                stop_group KILL
                signal_session KILL >/dev/null
                wait_gone 3
            fi
        fi
        wait "$cmd_pid" 2>/dev/null
    fi

    pattern_sweep
    remote_sweep
    if [ -n "$EXTRA_CLEAN" ]; then
        log "running extra cleanup: $EXTRA_CLEAN"
        bash -c "$EXTRA_CLEAN" </dev/null || log "extra cleanup returned $?"
    fi
    report_survivors

    # Guard last: if we are SIGKILLed during cleanup, it still finishes the job.
    [ -n "$guard_pid" ] && kill "$guard_pid" 2>/dev/null
    return 0
}

# Interruptible sleep: as a background job + wait, so that signals are
# processed immediately instead of only after the sleep has finished.
isleep() {
    sleep "$1" &
    sleep_pid=$!
    wait "$sleep_pid" 2>/dev/null
    sleep_pid=""
}

trap 'log "interrupted (SIGINT)"; cleanup; exit 130' INT
trap 'log "terminated (SIGTERM)"; cleanup; exit 143' TERM
trap 'log "hangup (SIGHUP)";      cleanup; exit 129' HUP
trap 'cleanup' EXIT

# --------------------------------------------------------------- start
# setsid -> separate session; use process substitution instead of a pipe
# so that $! is the PID of the command (with "cmd | tee" it would be tee's PID).
setsid "$@" > >(tee -a "$LOGFILE") 2>&1 &
cmd_pid=$!

cmd_pgid=$(proc_field "$cmd_pid" 3) || true
cmd_sid=$(proc_field "$cmd_pid" 4)  || true
: "${cmd_pgid:=$cmd_pid}"
: "${cmd_sid:=$cmd_pid}"
log "command PID $cmd_pid (PGID $cmd_pgid, SID $cmd_sid)"

# Dead man's switch: survives even 'kill -9' on the watchdog itself.
if [ "$GUARD" -eq 1 ]; then
    setsid bash -c "
        while kill -0 $$ 2>/dev/null; do sleep 5; done
        kill -TERM -- -$cmd_pgid 2>/dev/null
        sleep $GRACE
        kill -KILL -- -$cmd_pgid 2>/dev/null
    " >/dev/null 2>&1 &
    guard_pid=$!
fi

# --------------------------------------------------------------- watch loop
idle=0
last=$(size)
while kill -0 "$cmd_pid" 2>/dev/null; do
    isleep "$INTERVAL"
    cur=$(size)
    if [ "$cur" != "$last" ]; then
        last=$cur
        idle=0                                    # output received -> reset counter
        continue
    fi
    idle=$((idle + 1))
    kill -0 "$cmd_pid" 2>/dev/null || break       # terminated on its own in the meantime
    log "no new output ($idle/$MAX_IDLE)"
    if [ "$idle" -ge "$MAX_IDLE" ]; then
        log "$((INTERVAL * MAX_IDLE))s without output - terminating command"
        exit "$TIMEOUT_RC"                        # EXIT trap does the escalation
    fi
done

wait "$cmd_pid"
status=$?
log "command terminated (exit $status)"
exit "$status"
