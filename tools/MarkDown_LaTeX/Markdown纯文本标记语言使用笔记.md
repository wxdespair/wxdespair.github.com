## 常用语法

- 关于标题

  ```markdown
  标题能显示出文章的结构。
  行首插入1-6个 # ，每增加一个 # 表示更深入层次的内容，对应到标题的深度由 1-6 阶。
  H1 : # Header 1
  H2 : ## Header 2
  H3 : ### Header 3
  H4 : #### Header 4
  H5 : ##### Header 5
  H6 : ###### Header 6
  ```

- 关于文本样式

  - （带“*”星号的文本样式，在原版Markdown标准中不存在，但在其大部分衍生标准中被添加）
     * 链接 :[Title](https://www.baidu.com/s?ie=UTF-8&wd=markdown)
     - 加粗 : **Bold**
     - 斜体字 : *Italics*
     - 高亮 : ==text==
     - 段落 : 段落之间空一行
     - 换行符 : 一行结束时输入两个空格或者添加\后再接回车
     - 列表 : * 添加星号成为一个新的列表项。
     - 引用 : > 引用内容
     - 内嵌代码 : `print('Hello World')`
     - 画水平线 (HR) : --------
     - 方框： - [ ] -

- 关于图片

  -     基本语法

        - ![Alt text](图片链接 "optional title")
        - ![焰灵姬](yanlingji.jpg)    
        - Alt text：图片的Alt标签，用来描述图片的关键词，可以不写。最初的本意是当图片因为某种原因不能被显示时而出现的替代文字，后来又被用于SEO，可以方便搜索引擎根据Alt text里面的关键词搜索到图片。
        - 图片链接：可以是图片的本地地址（相对路径与绝对路径均可）或者是网址。
        - "optional title"：鼠标悬置于图片上会出现的标题文字，可以不写。
        
  - 高级用法——把图片存入markdown文件
    用base64转码工具把图片转成一段字符串，然后把字符串填到基础格式中链接的那个位置。

    -     基础用法：    

    -     ![avatar](data:image/png;base64,iVBORw0......)    

    - 这个时候会发现插入的这一长串字符串会把整个文章分割开，非常影响编写文章时的体验。如果能够把大段的base64字符串放在文章末尾，然后在文章中通过一个id来调用，文章就不会被分割的这么乱了。

      -     比如：
    
      -     ![avatar][base64str]
    
      -     [base64str]:data:image/png;base64,iVBORw0......使用下
    
      - 面的python代码可以得到图片的base64代码（而且有待考证）：
      - ```python 
        # 将图片转化为字符串
        import base64 # 但首先要安装base64库
        f=open('yanlingji.jpg','rb') # 二进制方式打开图文件
        ls_f=base64.b64encode(f.read()) # 读取文件内容，转换为base64编码
        f.close()
        print(ls_f)
        # 将字符串转化为图片
        import base64
        bs='iVBORw0KGgoAAAANSUhEUg....' # 太长了省略
        imgdata=base64.b64decode(bs)
        file=open('2.jpg','wb')
        file.write(imgdata)
        file.close()
        ```
  
- 关于脚注
  
  - 脚注不存在于标准Markdown中。    
  
  - 使用这样的占位符号可以将脚注添加到文本中:[^1]. 另外，你可以使用“n”而不是数字的[^n]所以你可以不必担心使用哪个号码。
  
  - 在您的文章的结尾，你可以如下图所示定义匹配的注脚，URL将变成链接:
  
    - 这里是脚注[^1]  [^1]: https://www.baidu.com/s?ie=UTF-8&wd=markdown
  
      这里是脚注[^n]   [^n]: https://www.baidu.com/s?ie=UTF-8&wd=latex
  
- 插入代码

  - 添加内嵌代码可以使用一对回勾号(tab上面的那个)
  - `print('Hello World')`
  - 对于插入代码, Ghost支持标准的Markdown代码和GitHub Flavored Markdown (GFM)。
  - 标准Markdown基于缩进代码行或者4个空格位，而GFM 使用三个回勾号```


<h1>{{title}}</h1>

```html
<h1>{{title}}</h1>
```



<如果这之间的文本为纯英文应该不会被显示，可是不知道后面的咋地了……它显示了> <like this>

- 常用的Markdown 编辑器
  - OSX平台：VSCode、Atom、Byword、Mou、Typora、MacDown、RStudio  
  - Linux平台下：VSCode、Atom、Typora、ReText、UberWriter、RStudio\
  - Windows平台下：VSCode、Atom、CuteMarkEd、MarkdownPad2、Miu、Typora、RStudio\
  - iOS平台下：Byword\
  - 浏览器插件：MaDo (Chrome)、Marxico（Chrome）\
  - 高级应用：Sublime Text 3 + MarkdownEditing / 教程





# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题

> 一级引用
> > 二级引用
> >
> > > 三级引用

* 一级列表
    * 二级列表
        * 三级列表

1. 有序列表
    1. 有序子列表
    
    * 有序子列表
* 有序列表
    * 无序子列表
    * 无序子列表

```
System.out.println("MarkDown");
```

`重点文字`

~~删除线~~

__下划线__

***粗斜体***

**粗体**

*斜体*

添加链接

[百度一下](https://www.baidu.com)

链接图片

![百度Logo](https://www.baidu.com/img/bd_logo.png)

这是个简单的表格

| First Header | Second Header | Third Header |
| ------------ | ------------- | ------------ |
| Content Cell | Content Cell  | Content Cell |
| Content Cell | Content Cell  | Content Cell |

出于美观的考虑，可以把两端都包围起来

| First Header | Second Header | Third Header |
| ------------ | ------------- | ------------ |
| Content Cell | Content Cell  | Content Cell |
| Content Cell | Content Cell  | Content Cell |

通过在标题分割行添加冒号`:`，你可以定义表格单元的对其格式：向左靠齐，居中和向右靠齐

| First Header | Second Header | Third Header |
| :----------- | :-----------: | -----------: |
| Left         |    Center     |        Right |
| Left         |    Center     |        Right |

这是一个示例的表格使用案例

| 名字 | 年龄 | 描述 |
| ---- | ---- | ---- |
|      |      |      |

简书上一篇介绍Markdown语法规则的文章  [点击跳转](https://www.jianshu.com/p/191d1e21f7ed)
