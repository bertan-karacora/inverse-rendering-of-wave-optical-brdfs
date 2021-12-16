from subprocess import check_call, CalledProcessError
import os

try:
    check_call(["apt", "install", "libeigen3-dev", "libopenexr-dev"], stdout=open(os.devnull,"wb"))
except CalledProcessError as e:
    print(e.output)
