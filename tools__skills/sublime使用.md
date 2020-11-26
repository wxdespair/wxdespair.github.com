ctrl键多点编辑

ctrl+P可以根据文件名快速打开文件

shift+ctrl+P 唤出命令面板

ctrl + +/-可以快速调节字体大小

调出命令面板，输入keybin进入Key Bindings，该文件存放了sublime的所有默认快捷键

几个快捷键：

- Ctrl + tab 切换不同文件的页面  
- Ctrl + j 合并一行，中间会自动添加一个空格 
- Ctrl + d 选中当前单词，继续敲可以选中多个，某种程度上相当于建立多个编辑点  
- Ctrl + / 注释掉/取消注释  
- Ctrl + ]/[ 缩进  
- Ctrl + L 选择当前行  
- Ctrl + enter / Shift + Ctrl + enter 当前行之前或之后开辟一行   
- 列选择，按住 Option 键，用鼠标左键拖动。(暂时没找到Mac OS中的Option键对应window中的那个键) 
- shift + ctrl + ↑/↓ 快速将当前行移动到前/后一行
- ctrl + ~ 可以快速调出操作台

## 操作粒度

同一行之上，直接用左右箭头，每次移动一个字符，Alt 加上箭头，每次就可以移动一个单词，如果配合上 Cmd 就一下到头了。 这个道理在选择的时候也管用，差别就是在加上 Shift 键，例如向左选择一个单词就是 Shift-Alt-Left 。

## 自定制代码片段

snippet 这个英文单词的意思就是，小片段。 sublime 可以让用户创建自己的 snippet 。

### 寻找重复操作

创建什么呢？这个不是凭空想出来的，而是要定位自己日常工作中的重复性劳动。比如我在写视频笔记的时候，用的时 github 的 jekyll 这种格式，每个页面头上都要有这样几行内容

```
---
layout: default
title: 标题
---
```

而且输入完之后还要输入一个空行，然后再来输入正文。根据这个需求，就可以 menu->tools->new snippet 来创建下面的内容

```xml
<snippet>
  <content><![CDATA[
---
layout: default
title: ${1:标题}
---

${2}
]]></content>
  <tabTrigger>header</tabTrigger>
  <scope>text.html.markdown</scope>
</snippet>
```

命名为 jekyll-header.sublime-snippet ，保存到 Packages/User 之下。这样到一个 markdown 文件中，命令面板中输入 `snippet` 就可以看到这个 snippet 的信息了。

敲 snippet 的同时也看到了 php 相关的 snippet 。但是其实我是不写 php 的，那怎么样避免这些信息出现呢？

### php snippet 如何卸载

其实这个问题就是，默认就装好的包，如何来禁用。到 Settings User 中

```json
"ignored_packages":[
  "Vintage", // missing this will trigger vintage mode
  "PHP", // to avoid php snippets in html files
],
```

### 作用范围 scope

最后来说说上面 snippet 中填写的 scope 是怎么得到的。

在 keybinding Default 中又这样的设置

```json
{ "keys": ["super+alt+p"], "command": "show_scope_name" },
{ "keys": ["ctrl+shift+p"], "command": "show_scope_name" },
```

所以可以通过上面的任意一组快捷键来读取当前文件的 scope ，爆出来的内容有可能是空格隔开的多个字符串，取第一个字符串就行了。基本上各种代码文件的 scope 都是 `source` 打头，例如 `source.js` `source.ruby` `source.python` 。其他的都是以 `text` 打头，例如 `text.html.basic` `text.html.markdown` `text.plain` 等。 那么如何给一个 snippet 定义多于一个文件类型的 scope 呢？可以这样

```xml
<scope>text.html.markdown, text.plain</scope>
```

 http://happypeter.github.io/happysublime/ 

## 批处理任务

来聊创建自己的 Build System。如果你发觉你总在输入重复的文字内容，那就要用前面的代码补全的功能。如果你发现有一系列操作或者是命令是要很频繁的执行的，例如一个软件写完之后要测试编译执行看报错，那就可以把这些操作写成一个 build，一键完成。

参考 [这里](http://sublime-text-unofficial-documentation.readthedocs.org/en/latest/reference/build_systems.html) 。

### chrome 打开 html 文件

名字虽然叫 build 但是执行的任务也不是非得编译软件，任何命令都可以呀，只要能自动化重复性劳动就行呗。

menu->tools->build system-> new build system 这里打开一个文件，粘贴下面内容

```json
{
  "cmd": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "$file"],
  "selector": "text.html"
}
```

注意，根据 [这里](http://sublime-text-unofficial-documentation.readthedocs.org/en/latest/reference/build_systems/configuration.html) 的说明，selector 生效的前提是 menu->tools-> build system->Automatic 设置为 true 。

保存到 User/ 之下，名字叫 browse.sublime-build 。

### build 我的 jekyll 页面

再来个稍微复杂点的。还是说我这里的视频笔记，每次写完一些内容之后，我都要执行

```
git commit -a -m"wip" && git push
```

把新改的内容做成一个版本然后再推送到 github 上面，这样我再到 github 上对应的页面刷新，就看到效果了。这些步骤不少，看看怎么样做成一个 build 来一键完成。

大体思路是这样，把所有的工作都写成一个 bash 脚本，然后在 build 这里直接执行。

jekyll.sublime-build 中这样写

```json
{
    "cmd": ["/Users/peter/bin/jekyll.sh", "$file"],
    "working_dir": "$file_path",
    "selector": "text.html.markdown"
}
```

注意 jekyll.sh 中一定要写 shebang 也就是第一行的声明，不然 sublime 中就会报格式错误，另外就是要执行

```
chmod +x jekyll.sh
```

关于命令行使用和 shell 脚本编程，可以参考 [Linux Guide for Developers](http://www.imooc.com/view/181) 这门课程。

现在，就可以执行 jekyll.sh 了，这个里面写

```sh
#!/usr/bin/env bash
git commit -a -m"wip" && git push
# echo  $1
VAR=$1 # full file patch, e.g /Users/peter/rails10-va/happysublime/10_build.md
FILE=`basename $VAR` # get 10_build.md from full path
PAGE=${FILE%.*}".html" # 10_build.md -> 10_build.html

DIR=`dirname $VAR`
PROJECT=`basename $DIR` # get happysublime
# echo $PARENT
URL="happypeter.github.io/"$PROJECT"/"$PAGE
'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' $URL
```

这样每次我修改完视频笔记就可以直接敲 Cmd-b 推送到 github 上，并且在 chrome 中打开看到效果了。不过美中不足的时并不是直接推送到 github 上的内容， jekyll 网站页面上就会立即生效，所以一般要等几秒再刷新一下才能看到效果。