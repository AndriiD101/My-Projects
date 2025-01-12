import subprocess

# Define the base command
base_command = [
    'python', 
    r'C:\Users\denys\Desktop\HOP\final_TS.py', 
    '-f', r'C:\Users\denys\Desktop\HOP\github_code\data_git.txt'
]

# Define ranges for -i and -s
n_values = ["a", "b", "c", "d", "e", "f"]
i_values = [0, 1, 5, 10, 50, 100, 150, 200]  # Example values for -i
s_values = [0, 3, 5, 10]         # Example values for -s

# Loop through combinations of -i and -s values
for n in n_values:
    print(f"NODE: {n}")
    for i in i_values:
        print(f"ITERATION: {i}")
        for s in s_values:
            # print(f"TABU SIZE: {s}")
            
            # Construct the full command
            command = base_command + ['-i', str(i), '-s', str(s), '-n', str(n)]

            try:
                # Run the command
                result = subprocess.run(command, capture_output=True, text=True, check=True)

                # Print the output only
                print(f"for node={n}, for iter={i}, for size={s}", result.stdout.strip())  # Print STDOUT without extra text
            except subprocess.CalledProcessError as e:
                # Print only error output if the command fails
                print(e.stderr.strip())  # Print STDERR without extra text
