## VISION model and Step Function

This senario performs model development using the Object Detection model of Yolov5 and then builds the MLOps architecture directly using AWS Step Functions from the customer's account. 
All implementations are directly done by the customer and the issues that arise during the course of the program are shared by the SAs in charge and supported by debugging.




![ref-architecture.png](./figures/ref-architecture.png)

![ref-architecture-implements.png](./figures/ref-architecture-implements.png)


## Known issues

#### ImportError: libGL.so.1: cannot open shared object file: No such file or directory

When using the pytorch inference image from the docker, this error might be occured. A solution to the issue can be found in the [inference docker file](./1.SageMaker-Training-Processing/docker/Dockerfile.inf). Please uncomment the line, below and run the docker build command.
````
# For gaining a tmp directory priviledge
# RUN chmod 777 -R /tmp
````
#### LAMBDA_RUNTIME Failed to get next invocation. Http Response code: 404

When using an inference image in a lambda function, this error might be occured. A potential fix could be adding the following lines to the [docker file](./3.MLOps-Approval-Evaluation/Dockerfile)
````
# To fix a python runtime version
ENV AWS_LAMBDA_RUNTIME_API=3.9
````

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

