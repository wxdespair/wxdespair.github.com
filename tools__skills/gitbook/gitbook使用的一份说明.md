gitbook使用实录

其实我从2年前已经接触了gitbook了，也使用这个工具编写电子书，但有几个技术问题一直无暇解决，拖到现在。趁国庆期间集中研究了，现在抽空做一些记录。

一、安装及运行
1.1 安装
gitbook使用npm安装，命令如下：

```
sudo npm install -g gitbook
sudo npm install -g gitbook-cli

```

如果提示错误信息UNABLE_TO_GET_ISSUER_CERT_LOCALLY，可能是ssl检查不通过，则设置：

```
npm config set strict-ssl false
```

1.2 安装插件
配置好之后，安装插件：

```
gitbook install
```

1.3 运行服务
在本地运行服务：

```
gitbook serve
```

默认端口为4000，使用浏览器访问http://127.0.0.1:4000即可。在服务运行期间，修改任何文件，会自动更新，不需要重启服务。此方式十分适合边编辑边看效果。

1.4 生成pdf文件
先安装插件：

```
npm install ebook-convert -g
sudo apt install calibre
```


再在图形界面执行：

```
gitbook pdf .
```

如果要指定pdf文件名称，则使用：

```
gitbook pdf . mybook.pdf
```

1.5 托管
生成静态网页：

```
gitbook build
```

成功后，将产生_book目录，该目录即为一个完整的静态网页(html格式文件)，将其放到web服务目录即可进行访问。我们可以使用CI系统进行自动化构建，将产生的静态网页发布到云服务器或github上。

二、配置
2.1 根目录文件
电子书根目录文件说明如下：

| 文件名        | 说明                                             |
| ------------- | ------------------------------------------------ |
| README.md     | 必须，电子书首页，一般做简单介绍，相当于“前言”   |
| SUMMARY.md    | 必须，大纲目录，里面指定了各章节的目录           |
| book.json     | 必须，配置文件，指定电子书标题、语言、插件等信息 |
| postscript.md | 可选文件，相当于“后记”                           |
| node_modules  | 执行gitbook install自动生成                      |
| _book         | 执行gitbook build自动生成                        |

2.2 章节设计
gitbook的电子书源文件使用Markdown编写，在着手前需要确定好大纲，一般到三级目录即可。

建议各章节使用ch01这样的形式命名，内部按小章节分别使用不同的.md文件编写，至于图片，则按章节划分，以章节目录下再创建同名目录存放图片文件，以img1.1.jpg、img2.1.jpg这样的格式命名。目录及文件示例如下：

```
.
├── book.json
├── ch01
│   ├── ch01 # 放到本章节图片文件
│   ├── ch01.md # 所有小章节在同一文件
│   └── README.md
├── ch02
│   ├── ch02 # 放到本章节图片文件
│   ├── ch02.1.md # 按小章节创建文件
│   ├── ch02.2.md
│   ├── ch02.3.md
│   ├── ch02.4.md
│   └── ch02.md
├── favicon.ico
├── postscript.md
├── README.md
├── readme.txt
├── res
│   ├── ch01
│   ├── ch02
│   └── 流程图.vsd
├── SUMMARY.md
└── wqy-zenhei.ttc # 字体文件
```

三、问题
3.1 打开生成的_book目录下html，但页面无法跳转
在该目录下，进入gitbook目录，找到目录下的theme.js文件，将if(m)改成if(false)。

3.2 页内跳转
在SUMMARY.md文件的目录中指定好锚点，如* [1.1 系统安装](ch01/ch01.md#ch11)。
然后在对应的md文件中写锚点，如<span id="ch11"></span>。注意，一定要单独一行，不能用"."、"_"这些符号。

3.3 图片居中
markdown没有快捷的方法设置图片居中，只能使用html标签，示例如下：

<div align=center>![](ch01/img1.1.png)
<center>图1.1 图片说明</center>
3.4 表格表格形式如下：

```
|  参数   |  说明   |
| ------ | ------ |
| 参数1 |  说明1
| 参数2 |  说明2
```

注意，前后一定要空一行，否则不生效。如果表格中有|的，要使用 &#124;符号，不能用反引号\|，因为会被解析为表格的格式|。另外如*、$、#等字符亦要特殊处理，如果要在表格中显示\#，则要填写\\#，否则\不会显示出来。

3.5 忽略文件
新建.bookignore文件，在该文件指定需要忽略的文件。示例如下：

```
readme.txt
LICENSE
*.back
tmp
```

3.6 大纲目录自动编号
gitbook支持大纲章节自动编号，在book.json文件中，在pluginsConfig添加配置：

```
"theme-default": {
            "showLevel": true
        },
```


同时添加插件"expandable-chapters-small@^0.1.7",。大纲示例：

```
# Summary

## [前言](README.md)

## [嵌入式Linux基础概念](ch01/README.md)
  * [嵌入式Linux是什么](ch01/ch01.md#ch11)
  * [嵌入式Linux由哪些部分组成](ch01/ch01.md#ch12)
  * [如何学习嵌入式Linux](ch01/ch01.md#ch13)
  * [其它知识点补充](ch01/ch01.md#ch14)
  * [本书代码示例说明](ch01/ch01.md#ch15)
  * [本书实验平台说明](ch01/ch01.md#ch16)
  * [本章小结](ch01/ch01.md#ch17)
```

如果手动编号，则在book.json中去掉插件"expandable-chapters-small@^0.1.7",，去掉showLevel。大纲示例：

```
# Summary

* [前言](README.md)

* [第1章 嵌入式Linux基础概念](ch01/README.md)
    * [1.1 嵌入式Linux是什么](ch01/ch01.md#ch11)
    * [1.2 嵌入式Linux由哪些部分组成](ch01/ch01.md#ch12)
    * [1.3 如何学习嵌入式Linux](ch01/ch01.md#ch13)
  * [1.4 其它知识点补充](ch01/ch01.md#ch14)
  * [1.5 本书代码示例说明](ch01/ch01.md#ch15)
  * [1.6 本书实验平台说明](ch01/ch01.md#ch16)
  * [1.7 本章小结](ch01/ch01.md#ch17)
```

两种编号方式效果如下图所示：

![20191011120019987](C:\Users\Administrator\Documents\document\20191011120019987.png)


从图中看出，自动编号从前言部分开始，虽然省去了逐个编号的麻烦，但机动性并不强，笔者最终选择手动编号的方式。

3.7 pdf中文乱码
该问题产生原因一般是系统不支持中文字符或所用的字体文件不合适。
在book.json中对字体进行配置：

    "fontSettings": {
      "theme": "white",
      "family": "wqy-zenhei",
      "size": 2
    },
    "plugins": [
      "wqy-zenhei"
    ]
再将系统字符集设置为UTF8，如：

```
export LANG='en_US.UTF-8'
export LC_ALL='en_US.UTF-8'
export LANGUAGE='en_US.UTF-8'
```

最后将字体文件拷贝到/usr/share/fonts/truetype目录。字体文件可借鉴Windows系统的，在C:\Windows\Fonts找到各式字体文件（以.ttc结尾文件），为避免版权问题，建议使用开源的字体文件，如文泉驿正黑（http://wenq.org/）。

如此，便可用gitbook pdf .生成正常的pdf文件了。

四、配置示例
4.1 配置文件book.json
book.json文件配置示例如下：

    {
        "title": "《我的电子书》 by 李迟",
        "description": "《《我的电子书》》",
        "author": "李迟 <li@latelee.org>",
        "output.name": "site",
        "language": "zh-hans",
        "root": ".",
        "links": {
            "sidebar": {
                "返回主页": "https://www.latelee.org",
            }
        },
        "styles": {
            "website": "styles/website.css",
            "ebook": "styles/ebook.css",    
            "pdf": "styles/pdf.css",
            "mobi": "styles/mobi.css",
            "epub": "styles/epub.css"
        },
        "plugins": [
            "github",
            "github-buttons",
            "favicon",
            "include-codeblock@^3.0.2",
            "splitter@^0.0.8",
            "tbfed-pagefooter@^0.0.1",
            "anchor-navigation-ex@0.1.8",
            "splitter",
            "donate"
        ],
    "pluginsConfig": {
        "github": {
            "url": "https://github.com/latelee/"
        },
        "include-codeblock": {
            "template": "ace",
            "unindent": true,
            "edit": true
        },
        "tbfed-pagefooter": {
            "copyright": "Copyright © Late Lee 2018-2019",
            "modify_label": " Last update: ",
            "modify_format": "YYYY-MM-DD HH:mm:ss"
        },
        "anchor-navigation-ex": {
            "isRewritePageTitle": false,
            "tocLevel1Icon": "fa fa-hand-o-right",
            "tocLevel2Icon": "fa fa-hand-o-right",
            "tocLevel3Icon": "fa fa-hand-o-right"
        },
        "favicon": {
            "shortcut": "favicon.ico",
            "bookmark": "favicon.ico"
        },
        "donate": {
            "wechat": "https://raw.githubusercontent.com/latelee/public/master/wepay_240.png",
            "alipay": "https://raw.githubusercontent.com/latelee/public/master/alipay_240.png",
            "title": "",
            "button": "欢迎捐赠",
            "alipayText": "支付宝",
            "wechatText": "微信"
            },
        "fontSettings": {
          "theme": "white",
          "family": "wqy-zenhei",
          "size": 2
        },
        "plugins": [
          "wqy-zenhei"
        ]
        }
    }

4.2 大纲文件SUMMARY.md

```
# Summary

* [前言](README.md)

----

## 第一部分 Linux系统使用篇
* [第1章 嵌入式Linux基础概念](ch01/README.md)
    * [1.1 嵌入式Linux是什么](ch01/ch01.md#ch11)
    * [1.2 嵌入式Linux由哪些部分组成](ch01/ch01.md#ch12)
    * [1.3 如何学习嵌入式Linux](ch01/ch01.md#ch13)
  * [1.4 其它知识点补充](ch01/ch01.md#ch14)
  * [1.5 本书代码示例说明](ch01/ch01.md#ch15)
  * [1.6 本书实验平台说明](ch01/ch01.md#ch16)
  * [1.7 本章小结](ch01/ch01.md#ch17)
```

4.3 CI脚本
笔者目前使用GitLab托管电子书源码（GitHub作为备用仓库，双备份），其内置了CI系统，.gitlab-ci.yml脚本包括了设置字符集，拷贝字体文件，安装gitbook，生成电子书静态网页，生成pdf文件，然后再提交到指定的web地址。脚本敏感字段均使用CI系统环境变量进行保护。完整内容如下：

```
send_job:
  before_script:
    - apt-get update -qq && apt-get install -y -qq software-properties-common
    - curl -sL https://deb.nodesource.com/setup_10.x | bash -
    - echo "install nodejs"
    - apt-get update -qq && apt-get install -y -qq nodejs calibre
    - export TZ='Asia/Shanghai' # 更改时区
    - export LANG='en_US.UTF-8'
    - export LC_ALL='en_US.UTF-8'
    - export LANGUAGE='en_US.UTF-8'
    - cp wqy-zenhei.ttc /usr/share/fonts/truetype # 拷贝字体文件
    - npm install gitbook -g
    - npm install gitbook-cli -g
    - npm install ebook-convert -g
    - gitbook install
  script:
    - gitbook build # 生成书籍
    - mv _book /tmp
    - gitbook pdf . ellp.pdf # 生成pdf电子书
    - mv ellp.pdf /tmp
  after_script:
    - cd /tmp/
    - git clone https://${github_you_dont_need_to_know}
    - cd ${mywebsite}
    - rm book/ep/ -rf
    - cp -a /tmp/_book/ book/ep
    - cp -a /tmp/ellp.pdf book/ep
    - git config user.name  "Late Lee"
    - git config user.email "li@latelee.org"
    - git add .
    - git add -u .
    - git commit -m "CI auto update ellp book"
    - git push --force --quiet "https://${TravisCIToken}@${github_you_dont_need_to_know}" master:master
  only:
    - master
```

笔者在Windows系统使用MinGW环境安装，使用Ubuntu子系统环境安装，但运行时都遇到权限问题，故作罢，转向Linux系统。如果条件允许，建议使用gitbook serve运行服务，虽说都是Markdown，但毕竟gitbook有部分格式与常用的Markdown略有不同。
本文涉及的示例，均从笔者实际使用，测试所得，具有一定实践参考价值，但示例仅供参考。