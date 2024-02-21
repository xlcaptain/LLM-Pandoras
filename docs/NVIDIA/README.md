# 安装 NVIDIA 驱动指南

本指南适用于 CentOS 系统上安装 NVIDIA 驱动程序。请按照以下步骤操作，确保您具有 root 权限或使用 `sudo` 来执行这些命令。

## 步骤 1: 关闭图形界面

为了安装 NVIDIA 驱动，需要先关闭图形界面。可以通过以下命令切换到命令行模式：

```bash
sudo systemctl isolate multi-user.target
```


## 步骤 2: 禁用 nouveau 驱动

NVIDIA 安装前需要禁用默认的 nouveau 驱动。通过创建或编辑 `/etc/modprobe.d/blacklist-nouveau.conf` 文件，并添加以下内容来实现：

```bash
blacklist nouveau
options nouveau modeset=0
```

## 步骤 3: 安装依赖

安装编译 NVIDIA 驱动所需的依赖项：
```bash
sudo yum install kernel-devel kernel-headers gcc make
```


## 步骤 4: 赋予执行权限

下载的 `.run` 文件需要执行权限：
```bash
chmod +x NVIDIA-Linux-x8664-535.154.05.run
```


按照安装程序的引导完成安装过程。通常情况下，接受默认选项即可，但可能需要根据您的系统配置作出相应选择。

## 步骤 6: 重启系统

安装完成后，重启系统以应用更改：
```bash
sudo reboot
```

要从命令行模式切回到图形界面，您可以使用以下命令：

```bash
sudo systemctl isolate graphical.target
```

## 注意

这些步骤适用于大多数 CentOS 版本，但具体步骤可能会根据您的系统配置和已安装的软件包略有不同。如果在安装过程中遇到问题，请检查 NVIDIA 官方文档或寻求社区支持。