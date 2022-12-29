
local imageFlag = std.extVar("imageFlag");

{
    version: "3.9",
    services: {
        dummy: {
            command: "echo 'Hello World!'"
        } + if imageFlag == "true" then {
            image: "ubuntu:latest"
        } else {
            build: {
                context: ".",
                dockerfile: "Dockerfile"
            }
        }
    }
}