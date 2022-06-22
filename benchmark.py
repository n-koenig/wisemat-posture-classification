import subprocess

# r = subprocess.Popen(['sudo', 'pinpoint', 'bash -c "source ./venv/bin/activate && python3 main.py"'])
# r = subprocess.Popen(['sudo', 'pinpoint', '--', 'bash -c "python3 main.py"'])
r = subprocess.Popen(['sudo', 'pinpoint', '--', 'bash', '-c', 'source ./venv/bin/activate && python3 main.py'])
# r = subprocess.Popen(['sudo', 'perf', 'stat', '-a', '-e"power/energy-cores/,power/energy-gpu/,power/energy-ram/,power/energy-pkg/"', 'bash', '-c', 'source ./venv/bin/activate && python3 main.py'])
# r = subprocess.Popen(['sudo', 'perf', 'stat', '-a', '-e"power/energy-cores/,power/energy-gpu/,power/energy-ram/,power/energy-pkg/"', 'ls'])
# r = subprocess.Popen('sudo pinpoint -- bash -c "source ./venv/bin/activate && python3 main.py"')
output, errs = r.communicate()
print(output)