import subprocess
import statistics

reps = 20

# r = subprocess.Popen(['sudo', 'pinpoint', 'bash -c "source ./venv/bin/activate && python3 main.py"'])
# r = subprocess.Popen(['sudo', 'pinpoint', '--', 'bash -c "python3 main.py"'])
# r = subprocess.Popen(['sudo', 'perf', 'stat', '-a', '-e"power/energy-cores/,power/energy-gpu/,power/energy-ram/,power/energy-pkg/"', 'bash', '-c', 'source ./venv/bin/activate && python3 main.py'])
# r = subprocess.Popen(['sudo', 'perf', 'stat', '-a', '-e"power/energy-cores/,power/energy-gpu/,power/energy-ram/,power/energy-pkg/"', 'ls'])
# r = subprocess.Popen('sudo pinpoint -- bash -c "source ./venv/bin/activate && python3 main.py"')
# sudo pinpoint -r 1 -c -- bash -c source ./venv/bin/activate && python3 main.py
# r = subprocess.Popen(['sudo', 'pinpoint', '-r', str(reps), '--', 'bash', '-c', 'source ./venv/bin/activate && python3 main.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
r = subprocess.Popen(['sudo', 'pinpoint', '-r', str(reps), '--', '--header', '-c', 'bash', '-c', 'source ./venv/bin/activate && python3 main.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
output, errs = r.communicate()
print(output)

f1_scores = [float(x) for x in output.decode().split("\n")[:-1]]
print(f1_scores)
print(statistics.mean(f1_scores))


