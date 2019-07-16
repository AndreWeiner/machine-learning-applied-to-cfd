docker container run                                      \
  -it -v=$PWD:/home                                       \
  --workdir=/home                                         \
  --user=$(id -u)                                         \
  -e USER=$USER                                           \
  --volume="/etc/group:/etc/group:ro"                     \
  --volume="/etc/passwd:/etc/passwd:ro"                   \
  --volume="/etc/shadow:/etc/shadow:ro"                   \
  --volume="/etc/sudoers.d:/etc/sudoers.d:ro"             \
  andreweiner/of_pytorch:of1906-py1.1-cpu                 \
  /bin/bash
