#### flowchart教程

言归正传，下面开始教程。其实捏，Flowchart是老外发明的，当然英文资料介绍得最确切，我也是看这些第一手资料学习过来的，金玉在前，这里我主要做一些引用和翻译的工作。

> 主要参考资料：[creately Ultimate Flowchart Guide](https://link.zhihu.com/?target=http%3A//creately.com/blog/diagrams/flowchart-guide-flowchart-tutorial/)

**Flowcharts的历史**

Flowcharts是由Frank Gilberth在1921年最先提出的，最开始的全名是”Process Flow Charts”，即处理流程图表。但是真正把Flowcharts推广开来的是这位叫Allan的老先生，他向商人培训Flowcharts的使用并逐渐流传开来。更多的历史详情可以阅读[维基百科](https://link.zhihu.com/?target=http%3A//en.wikipedia.org/wiki/Flowchart%23History)

**Flowchart符号的含义**

在开始画flowchart流程图之前你必须清楚地理解不同符号的含义，才能正确地使用它们。大多数人只懂得使用几个基本的符号，比如流程框和判断框，但是捏，还有许多其它的符号，正确地使用可以让你的流程图表达地更加准确。概览如下，再分别逐个说明。

**开始/结束符**
凡事有始有终，这个椭圆符号代表流程的开始或结束。

**处理框**
表示一个处理流程，表达方式为：动词 + 名词，比如：**编辑**视频，**提交**申请，**发送**给客户等。

**数据(I/O)**
表示数据对象，用平行四边形表示，一般作为处理框的输入或输出。

**判定/条件**
一个判断条件，在程序流程图中很常用，就是if else啦，用菱形表示。

**子流程/预定义流程**
如果一个处理太过复杂，不好在当前图中详细展示，则引用一个子流程，然后用一个图专门定义这个子流程，类似于程序中的函数定义。下图中Quality Procedure太过复杂，所以用子流程表示，在别处会专门定义。

**存储数据**

表示将数据存储在如硬盘、内存或其它存储设备中，我通常使用这个图标表示存储到数据库。

> 先介绍这么多吧，如果还想了解更冷门的符号，可以看看这里 [flowchart symbols](https://link.zhihu.com/?target=http%3A//creately.com/diagram-type/objects/flowcharts)

使用ProcessOn画Flowchart

用ProcessOn画图是很简单滴，学习曲线超低啊，如果你掌握了我上面介绍的基本流程图符号，那么剩下的工作就是拖拖拽拽了。还是简单说一下。

1. 首先新建一个[Flowchart文件](https://link.zhihu.com/?target=https%3A//www.processon.com/diagrams/new)，如下图
2. 选择一个模板，然后照猫画虎吧，对于新手可以练习画画下面这个流程图
3. tips：可以直接从前一个符号拉一个箭头，此时会弹出下一个符号供选择，不用每次都从左栏拖过来，如下图
4. enjoy yourself!

好了，入门篇先介绍到这里，作者有空再更新高级篇。

## [原网址]( https://zhuanlan.zhihu.com/p/22602497 )

例子：

```flow
st=>start: 用户登陆
op=>operation: 登陆操作
cond=>condition: 登陆成功 Yes or No?
e=>end: 进入后台

st->op->cond
cond(yes)->e
cond(no)->op
```



