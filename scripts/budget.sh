#!/usr/bin/env bash
# Fails when a module holds more code lines than budget.txt allows.
# A code line is non-blank and not a `//` comment; a file stops counting at `#[cfg(test)]`.
set -euo pipefail
cd "$(dirname "$0")/.."

count() {
    find "$@" -name '*.rs' -print0 | xargs -0 awk '
        FNR == 1 { skip = 0 }
        /^#\[cfg\(test\)\]/ { skip = 1 }
        !skip && !/^[[:space:]]*(\/\/|$)/ { n++ }
        END { print n + 0 }'
}

status=0
while read -r limit paths; do
    [[ -z "$limit" || "$limit" == \#* ]] && continue
    # shellcheck disable=SC2086 # paths is a space-separated list on purpose
    n=$(count $paths)
    printf '%5d / %-5d %s\n' "$n" "$limit" "$paths"
    if (( n > limit )); then
        echo "over budget: $paths" >&2
        status=1
    fi
done < budget.txt
exit "$status"
