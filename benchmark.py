import subprocess
import statistics

reps = 2

# r = subprocess.Popen(['sudo', 'pinpoint', 'bash -c "source ./venv/bin/activate && python3 main.py"'])
# r = subprocess.Popen(['sudo', 'pinpoint', '--', 'bash -c "python3 main.py"'])
# r = subprocess.Popen(['sudo', 'perf', 'stat', '-a', '-e"power/energy-cores/,power/energy-gpu/,power/energy-ram/,power/energy-pkg/"', 'bash', '-c', 'source ./venv/bin/activate && python3 main.py'])
# r = subprocess.Popen(['sudo', 'perf', 'stat', '-a', '-e"power/energy-cores/,power/energy-gpu/,power/energy-ram/,power/energy-pkg/"', 'ls'])
# r = subprocess.Popen('sudo pinpoint -- bash -c "source ./venv/bin/activate && python3 main.py"')
# sudo pinpoint -r 1 -c --header -- bash -c "source ./venv/bin/activate && python3 main.py"
# r = subprocess.Popen(['sudo', 'pinpoint', '-r', str(reps), '--', 'bash', '-c', 'source ./venv/bin/activate && python3 main.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)





with open("watt4.txt", "w") as f:
	command = ['sudo', 'pinpoint', '-r', str(reps), '-c', '--header', '-b', '4000', '-a', '4000', '--', 'bash', '-c', 'source ./venv/bin/activate && python3 main.py']
	r = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=f)
	output, errs = r.communicate()
	print(output)


# with open("watt2.txt", "r") as f:
# 	arr = np.loadtxt("watts.txt", skiprows=1, delimiter=',')
# 	# print(arr)
# 	plt.plot(arr)
# 	plt.show()	

# f1_scores = [float(x) for x in output.decode().split("\n")[:-1]]
# print(f1_scores)
# print(statistics.mean(f1_scores))


