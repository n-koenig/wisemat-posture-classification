sudo perf stat -r 2 -a -e"power/energy-cores/,power/energy-gpu/,power/energy-ram/,power/energy-pkg/" bash -c "source ./venv/bin/activate && python3 main.py"

acc: 74.40656791004818
f1: 0.7640141505482528
            
acc: 81.6259146885597
f1: 0.821757959415688

          2.568,14 Joules power/energy-cores/                                           ( +-  1,78% )
              6,45 Joules power/energy-gpu/                                             ( +-  3,73% )
            548,93 Joules power/energy-ram/                                             ( +-  0,84% )
          3.066,58 Joules power/energy-pkg/                                             ( +-  1,35% )

            313,96 +- 3,06 seconds time elapsed  ( +-  0,98% )

