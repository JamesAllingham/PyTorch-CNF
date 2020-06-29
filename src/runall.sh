#!/bin/bash

srun -J 1_2 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_1_2.out --error=test_1_2.err --time=08:00:00 python fig4.py --energy_fun=1 --width=2 --epochs=10000 --train &
srun -J 2_2 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_2_2.out --error=test_2_2.err --time=08:00:00 python fig4.py --energy_fun=2 --width=2 --epochs=10000 --train &
srun -J 3_2 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_3_2.out --error=test_3_2.err --time=08:00:00 python fig4.py --energy_fun=3 --width=2 --epochs=10000 --train &
srun -J 4_2 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_4_2.out --error=test_4_2.err --time=08:00:00 python fig4.py --energy_fun=4 --width=2 --epochs=10000 --train &
srun -J 1_8 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_1_8.out --error=test_1_8.err --time=08:00:00 python fig4.py --energy_fun=1 --width=8 --epochs=10000 --train &
srun -J 2_8 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_2_8.out --error=test_2_8.err --time=08:00:00 python fig4.py --energy_fun=2 --width=8 --epochs=10000 --train &
srun -J 3_8 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_3_8.out --error=test_3_8.err --time=08:00:00 python fig4.py --energy_fun=3 --width=8 --epochs=10000 --train &
srun -J 4_8 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_4_8.out --error=test_4_8.err --time=08:00:00 python fig4.py --energy_fun=4 --width=8 --epochs=10000 --train &
srun -J 1_32 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_1_32.out --error=test_1_32.err --time=08:00:00 python fig4.py --energy_fun=1 --width=32 --epochs=10000 --train &
srun -J 2_32 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_2_32.out --error=test_2_32.err --time=08:00:00 python fig4.py --energy_fun=2 --width=32 --epochs=10000 --train &
srun -J 3_32 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_3_32.out --error=test_3_32.err --time=08:00:00 python fig4.py --energy_fun=3 --width=32  --epochs=10000 --train &
srun -J 4_32 -A JMH233-SL3-GPU -p pascal --gres=gpu:1 --output=test_4_32.out --error=test_4_32.err --time=08:00:00 python fig4.py --energy_fun=4 --width=32  --epochs=10000 --train &