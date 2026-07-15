#!/usr/bin/env bash
LC_ALL=C

local_branch="$(git rev-parse --abbrev-ref HEAD)"

valid_branch_regex="^(feat|fix|chore|refactor|docs)\/[a-z0-9]+(-[a-z0-9]+)*$"

message=$(cat <<EOF
There is something wrong with your branch name.
Branch names in this project must adhere to this regex: $valid_branch_regex.
You should rename your branch to a valid name and try again.

Rename your branch with the following command:

git branch -m <new-branch-name>
EOF
)
if [[ ! $local_branch =~ $valid_branch_regex ]]
then
    echo "$message"
    exit 1
fi

exit 0
