 

gitbook init

 打开一个文件夹 MyGitBook，使用 `gitbook init` 初始化文件夹，会自动生成两个必要的文件 README.md 和 SUMMARY.md 

- **README.md**:  书的介绍文字，如前言、简介，在章节中也可做为章节的简介。
-  **SUMMARY.md**: 定制书籍的章节结构和顺序。

README.md 和 SUMMARY.md 是 GitBook 制作电子书的必要文件，可用 gitbook init 命令自动生成

GitBook 使用 SUMMARY.md 文件作为书籍的目录结构，可以用来制作书籍目录。

```markdown
// SUMMARY.md

# Summary
* [Introduction](README.md)
* Part I
    * [从命令行进行测试](Chapter1/CommandLine.md)
    * [Monkey](Chapter1/Monkey.md)
    * [monkeyrunner 参考](Chapter1/MonkeyrunnerReference.md)
        * [概览](Chapter1/MonkeyrunnerSummary.md)
        * [MonkeyDevice](Chapter1/MonkeyDevice.md)
        * [MonkeyImage](Chapter1/MonkeyImage.md)
        * [MonkeyRunner](Chapter1/MonkeyRunner.md)
* Part II
    * [Introduction](Chapter2/c1.md)
    * [Introduction](Chapter2/c2.md)
    * [Introduction](Chapter2/c3.md)
    * [Introduction](Chapter2/c4.md)
```



gitbook serve

 执行命令 gitbook serve ，gitbook 会启动一个 4000 端口用于预览 ,你可以你的浏览器中打开这个网址： [http://localhost:4000](http://localhost:4000/) 预览电子书效果 

 第二种预览方式，运行 gitbook build 命令后会在书籍的文件夹中生成一个 _book 文件夹, 里面的内容即为生成的 html 文件. 我们可以使用下面命令来生成网页而不开启服务器 

http://localhost:4000

### GitBook 插件

当遇到「左侧的目录折叠」这种需求的时候，就用到 GitBook 插件了。

官方获取插件地址： https://plugins.gitbook.com/

#### 安装插件

安装插件只需要在书籍目录下增加 `book.json` 文件，例如增加[ 折叠目录 ](https://plugins.gitbook.com/plugin/expandable-chapters-small)的插件，需要在 book.json 内增加下面代码:

```json
{
    "plugins": ["expandable-chapters-small"],
    "pluginsConfig": {
        "expandable-chapters-small":{}
    }
}
```

```json
{
    "plugins": ["mathjax"]
}
```

然后在工作目录终端执行 `gitbook install ./` 来安装插件即可：

```powershell
gitbook install ./
```







### 1、GitBook 安装

- [安装 Node.JS 环境](https://nodejs.org/zh-cn/)

- 安装 GitBook 环境

   直接终端里输入下面代码，直接最高权限安装，会提示输入电脑密码，输入即可

  ```
  sudo npm install gitbook-cli -g
  ```

  > 注意`sudo`是终端里强制最高权限执行的意思，因为经常会出现权限不够导致错误无法继续进行，所以需要经常用到这个命令。

### 2、建立电子书

1. 按视频中先在 Finder 里任何你想的位置建好电子书的主文件夹，然后在终端先打上`cd`（注意 cd 后面有空格），然后把建好的文件夹托到终端里，会自动出来文件夹路径，敲回车进入

2. 输入命令

   ```
   sudo gitbook init
   ```

    生成电子书，同时文件夹里生成2个文件 

   > **README.md** 电子书的介绍页
   >  **SUMMARY.md** 电子书的目录

3. 输入命令

   ```
   sudo gitbook serve
   ```

   生成本地文件并开启在线服务 

   > 本机预览地址：http://localhost:4000

------

- 备注：[视频里我用到的文章地址](https://www.jianshu.com/p/e86c702578df) 
- 如果装2个以上的电子书，需要设置不同的端口（[传送门](https://blog.csdn.net/moxiaomomo/article/details/53026645)）

------

> **Gitbook 结尾**
>  其实到这里，书已经建好了，下一步就是怎么把文件弄到网站里，让局域网的小伙伴和 BOSS 方便的查看。
>  至于怎么使用怎么编辑还是看视频更详细，但是MarkDown 语法这个技能，建议大家都学学，本身毫无难度，但确是写文章的好东西，自行知乎百度吧，文章非常多，这里略过。

### 3、本地WEB环境部署

这里我偷懒一下，因为部署本地局域网网站网上教程很多，我就不写了，本身视频里讲的很详细，如果视频里提到的教程文章链接在下面：





[gitbook安装与使用（含常用插件和book.json配置详解）]( https://blog.csdn.net/fghsfeyhdf/article/details/88403548 )