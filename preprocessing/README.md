#### N4 Bias Field correction implementation

Prerequsites: ensure [docker](https://www.docker.com/) is installed on your local machine

`git clone https://github.com/jamesowler/final-year-project.git .`

`cd code-repo/preprocessing`

`docker build -t python-n4:0.1 .`

`docker run --rm -u $(id -u):$(id -g) -v </path/to/image/directory>:/data python-n4:0.1 /data/<image-file-name.tif> -n4`
