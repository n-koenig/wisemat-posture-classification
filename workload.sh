sudo ln -s ~/.../pinpoint/build/pinpoint /usr/local/bin

sudo pinpoint -- bash -c "source ./venv/bin/activate && python3 main.py"

sudo perf stat -r 1 -a -e"power/energy-cores/,power/energy-gpu/,power/energy-ram/,power/energy-pkg/" bash -c "source ./venv/bin/activate && python3 main.py"
