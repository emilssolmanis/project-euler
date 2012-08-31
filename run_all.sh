#!/bin/sh
for (( problem=1 ; problem < 44 ; problem++ )); 
do
    python3 -c "import project_euler; print($problem, project_euler.problem_$problem())"
done
