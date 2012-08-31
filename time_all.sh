#!/bin/sh
if [ -z $@ ];
then
    for (( problem=1 ; problem < 44 ; problem++ )); 
    do
        elapsed=$(/usr/bin/time -f '%E' python3 -c "import project_euler; project_euler.problem_$problem()" 2>&1)
        echo "$problem $elapsed"
    done
else
    elapsed=$(/usr/bin/time -f '%E' python3 -c "import project_euler; project_euler.problem_$@()" 2>&1)
    echo "$@ $elapsed"
fi
