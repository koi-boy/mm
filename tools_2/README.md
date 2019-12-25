# adc-on-ideepviewer

## 1. 模块module_ideepviewer
该部分直接从mmdetection中提取，对应的版本信息为v1.0rc0 master
sha：3dc9ddb73315f016d782fdeb824ec1f5ef2e9f81

mmdetection:
<https://github.com/open-mmlab/mmdetection>

裸机基础环境安装参考[INSTALL.md](https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md)：

### 1.1 Docker使用帮助

建议直接采用Docker使用，相关版本已经配置完成，详见FTP服务器：

```
IP：  ftp://182.150.44.163:1021 
user：ADC_share 
pass：ADC_share#21@

内网地址是  ftp://172.27.11.61
```

使用方法参考：

下面是针对centos_mmdet_v1.0rc0_ssh_190925.tar.gz镜像写的文档，
可以直接使用centos_ideepviewer_1.1.tar.gz

- 1、服务器需要安装Docker CE，具体安装方法参考Docker官方文档。

    * Centos：<https://docs.docker.com/install/linux/docker-ce/centos/>
    * Ubuntu：<https://docs.docker.com/install/linux/docker-ce/ubuntu/>
    
    推荐使用Centos系统
    针对Docker19.03，添加nvidia-container-toolkit，参考地址：
    <https://github.com/NVIDIA/nvidia-docker>

- 2、基于共享文件centos_mmdet_v1.0rc0_ssh_190925.tar.gz 加载Docker镜像

    ```
    cd 文件保存目录
    docker load < centos_mmdet_v1.0rc0_ssh_190925.tar.gz
    ```

- 3、显示所有Docker镜像
    ```
    docker images
    ```
    应该出现 centos_mmdet_v1.0rc0   latest

- 4、基于当前镜像运行Docker容器
    ```
    docker run --gpus all --name centos_mmdet_ssh -itd -p 2222:22 --privileged=true --ipc=host centos_mmdet_v1.0rc0:latest /usr/sbin/init
    ```
    将端口2222映射到22，便于后面使用ssh进行连接
    
    ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
    建议在第4步运行过程中添加 --ipc=host，即


- 5、查看所有在运行的容器
    ```
    docker ps
    ```
    应该可以看到对应images的镜像容器

- 6、至此，可以登录docker容器进行模型的训练。
    ```
    docker exec -it centos_mmdet_ssh /bin/bash
    ```
    或者直接使用xshell等工具进行连接，其中ip为服务器ip地址，端口为上述的2222. 用户名root，密码111111


- 7、如果想使用Pycharm直接连接Docker容器进行代码的运行和调试，请参考总结：

    <https://www.jianshu.com/p/91841f87db09>

## 2. 项目使用说明

项目将ADC模型相关的工作进行拆分，分别包括如下部分：

- 1、Configs：包含各种模型的配置文件
- 2、data_explore: 包含前期对数据的初步探索，如不同Code的分布，bbox大小分布等
- 3、preprocess：包括训练之前对数据的各种处理操作，包括数据打标格式转换、标签重命名、数据增强、数据train/test拆分等。
- 4、training：模型训练部分，对训练过程进行简单封装，包括非分布式训练和分布式训练
- 5、testing：对训练部分输出的模型进行测试，输出以目标检测为标准的评价结果和所有信息，如每张图片上的bbox信息以及类别信息。
- 6、metrics: 对testing输出的结果进行按业务逻辑进行precision和recall的评价，包括输出confusion matrix等，同时对分错样本进行梳理整合放到对应目录。
- 7、utils: 项目使用的通用方法，如绘图、图像扩展名支持等。
- 8、work_flow: 对上述各种过程进行整合，一次性完成数据探索、数据转换、模型训练、模型评估等任务，初步实现模型的第一个版本。之后可针对每个模块拆分使用进行模型的进一步迭代优化。

最开始建议通过work_flow进行使用。

如有各种Bug，可修改并采用Git方式进行提交即可。