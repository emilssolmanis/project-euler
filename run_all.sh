#!/bin/sh
if [ -z $@ ];
then
    problems=$(grep problem_ project_euler.py | egrep -o '[[:digit:]]*' | sort -n | tail -n1)
    for (( problem=1 ; problem <= $problems ; problem++ )); 
    do
        answer=$(python3 -c "import project_euler; print(project_euler.problem_$problem())" 2>&1)
        echo "$problem $answer"
    done
else
    answer=$(python3 -c "import project_euler; print(project_euler.problem_$@())" 2>&1)
    echo "$@ $answer"
fi
