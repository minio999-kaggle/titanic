(docker stop "ds-jupyter") 2> $null

docker run `
    --name "ds-jupyter" `
    --rm `
    -p 8888:8888 `
    -v "${PWD}:/home/jovyan" `
    -e JUPYTER_ENABLE_LAB=yes `
    -e GRANT_SUDO=yes `
    --user root `
    jupyter/datascience-notebook:ubuntu-20.04