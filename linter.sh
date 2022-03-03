#!/bin/bash -e
declare threshold=${1:-'9.00'}
declare path=${2:-'./src'}

pip install pylint==2.10.2
sudo apt-get update 
sudo apt-get install -y bc 

declare result=$( pylint --good-names=df "${path}" )
echo "${result}"
declare score=$( echo ${result} | grep "Your code has been rated at" | sed -n 's@^.*at @@p' | sed -n 's@/.*@@p' )
echo "YOUR SCORE FROM PYLINT: ${score}"
if [[ $( echo "${score}>${threshold}" | bc -l ) == "1" ]]; then
    echo "Linting successful. Your score was: ${score}. The threshold is: ${threshold}"
else 
    echo "Failure. Your score was: ${score}. The threshold is: ${threshold}" && exit 1 
fi