# adc-on-ideepviewer

## 1. ģ��module_ideepviewer
�ò���ֱ�Ӵ�mmdetection����ȡ����Ӧ�İ汾��ϢΪv1.0rc0 master
sha��3dc9ddb73315f016d782fdeb824ec1f5ef2e9f81

mmdetection:
<https://github.com/open-mmlab/mmdetection>

�������������װ�ο�[INSTALL.md](https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md)��

### 1.1 Dockerʹ�ð���

����ֱ�Ӳ���Dockerʹ�ã���ذ汾�Ѿ�������ɣ����FTP��������

```
IP��  ftp://182.150.44.163:1021 
user��ADC_share 
pass��ADC_share#21@

������ַ��  ftp://172.27.11.61
```

ʹ�÷����ο���

���������centos_mmdet_v1.0rc0_ssh_190925.tar.gz����д���ĵ���
����ֱ��ʹ��centos_ideepviewer_1.1.tar.gz

- 1����������Ҫ��װDocker CE�����尲װ�����ο�Docker�ٷ��ĵ���

    * Centos��<https://docs.docker.com/install/linux/docker-ce/centos/>
    * Ubuntu��<https://docs.docker.com/install/linux/docker-ce/ubuntu/>
    
    �Ƽ�ʹ��Centosϵͳ
    ���Docker19.03�����nvidia-container-toolkit���ο���ַ��
    <https://github.com/NVIDIA/nvidia-docker>

- 2�����ڹ����ļ�centos_mmdet_v1.0rc0_ssh_190925.tar.gz ����Docker����

    ```
    cd �ļ�����Ŀ¼
    docker load < centos_mmdet_v1.0rc0_ssh_190925.tar.gz
    ```

- 3����ʾ����Docker����
    ```
    docker images
    ```
    Ӧ�ó��� centos_mmdet_v1.0rc0   latest

- 4�����ڵ�ǰ��������Docker����
    ```
    docker run --gpus all --name centos_mmdet_ssh -itd -p 2222:22 --privileged=true --ipc=host centos_mmdet_v1.0rc0:latest /usr/sbin/init
    ```
    ���˿�2222ӳ�䵽22�����ں���ʹ��ssh��������
    
    ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
    �����ڵ�4�����й�������� --ipc=host����


- 5���鿴���������е�����
    ```
    docker ps
    ```
    Ӧ�ÿ��Կ�����Ӧimages�ľ�������

- 6�����ˣ����Ե�¼docker��������ģ�͵�ѵ����
    ```
    docker exec -it centos_mmdet_ssh /bin/bash
    ```
    ����ֱ��ʹ��xshell�ȹ��߽������ӣ�����ipΪ������ip��ַ���˿�Ϊ������2222. �û���root������111111


- 7�������ʹ��Pycharmֱ������Docker�������д�������к͵��ԣ���ο��ܽ᣺

    <https://www.jianshu.com/p/91841f87db09>

## 2. ��Ŀʹ��˵��

��Ŀ��ADCģ����صĹ������в�֣��ֱ�������²��֣�

- 1��Configs����������ģ�͵������ļ�
- 2��data_explore: ����ǰ�ڶ����ݵĳ���̽�����粻ͬCode�ķֲ���bbox��С�ֲ���
- 3��preprocess������ѵ��֮ǰ�����ݵĸ��ִ���������������ݴ���ʽת������ǩ��������������ǿ������train/test��ֵȡ�
- 4��training��ģ��ѵ�����֣���ѵ�����̽��м򵥷�װ�������Ƿֲ�ʽѵ���ͷֲ�ʽѵ��
- 5��testing����ѵ�����������ģ�ͽ��в��ԣ������Ŀ����Ϊ��׼�����۽����������Ϣ����ÿ��ͼƬ�ϵ�bbox��Ϣ�Լ������Ϣ��
- 6��metrics: ��testing����Ľ�����а�ҵ���߼�����precision��recall�����ۣ��������confusion matrix�ȣ�ͬʱ�Էִ����������������Ϸŵ���ӦĿ¼��
- 7��utils: ��Ŀʹ�õ�ͨ�÷��������ͼ��ͼ����չ��֧�ֵȡ�
- 8��work_flow: ���������ֹ��̽������ϣ�һ�����������̽��������ת����ģ��ѵ����ģ�����������񣬳���ʵ��ģ�͵ĵ�һ���汾��֮������ÿ��ģ����ʹ�ý���ģ�͵Ľ�һ�������Ż���

�ʼ����ͨ��work_flow����ʹ�á�

���и���Bug�����޸Ĳ�����Git��ʽ�����ύ���ɡ�