nvidia-smi
make remake -f makefile.linux -C WaveOpticsBrdfWithPybind/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bertan/Bachelorarbeit/Inverse-Rendering-of-Wave-Optical-BRDFs/extern/enoki/build/enoki:/home/bertan/Bachelorarbeit/Inverse-Rendering-of-Wave-Optical-BRDFs/extern/enoki/build
export LD_LIBRARY_PATH
python3 main.py $@
