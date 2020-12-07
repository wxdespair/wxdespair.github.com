## LaTeX编辑数学公式基本语法元素

LaTeX中的数学模式有两种形式：inline 和 display。前者是指在正文插入行间数学公式,后者独立排列,可以有或没有编号。

- **行间公式**(inline):用`$...$`将公式括起来。
- **块间公式**(displayed)，用`$$...$$`将公式括起来是无编号的形式，还有`\[...\]`的无编号独立公式形式，但Markdown好像不支持。块间元素默认是居中显示的。

## **各类希腊字母编辑表**

 常用的包括：`\alpha`, `\beta`,`\omega`分别为α,β,ω. 大写字母`\Theta`, `\Gamma`,`\Omega`为 Θ ，Γ，Ω. 

---

<center>各类希腊字母表</center>
| LaTeX 显示    | 语句        | LaTeX 显示  | 语句      | LaTeX 显示  | 语句      | LaTeX 显示 | 语句     |
| ------------- | ----------- | ----------- | --------- | ----------- | --------- | ---------- | -------- |
| $\alpha$      | \alpha      | $\theta$    | \theta    | $o$         | o         | $\tau$     | \tau     |
| $\beta$       | \beta       | $\vartheta$ | \vartheta | $\pi$       | \pi       | $\upsilon$ | \upsilon |
| $\gamma$      | \gamma      | $\iota$     | \iota     | $\varpi$    | \varpi    | $\phi$     | \phi     |
| $\delta$      | \delta      | $\kappa$    | \kappa    | $\rho$      | \rho      | $\varphi$  | \varphi  |
| $\epsilon$    | \epsilon    | $\lambda$   | \lambda   | $\varrho$   | \varrho   | $\chi$     | \chi     |
| $\varepsilon$ | \varepsilon | $\mu$       | \mu       | $\sigma$    | \sigma    | $\psi$     | \psi     |
| $\zeta$       | \zeta       | $\nu$       | \nu       | $\varsigma$ | \varsigma | $\omega$   | \omega   |
| $\eta$        | \eta        | $\xi$       | \xi       |             |           |            |          |
| $\Gamma$      | \Gamma      | $\Lambda$   | \Lambda   | $\Sigma$    | \Sigma    | $\Psi$     | \Psi     |
| $\Delta$      | \Delta      | $\Xi$       | \Xi       | $\Upsilon$  | \Upsilon  | $\Omega$   | \Omega   |
| $\Theta$      | \Theta      | $\Pi$       | \Pi       | $\Phi$      | \Phi      |            |          |

---

## **上下标、根号与省略号**

* 下标： _ 	     eg：$x_i$

* 上标： ^          eg：$x^2$
  
  注意：上下标如果多于一个字母或符号，需要用一个 {} 括起来  eg：$x_{i1}$   $x^{\alpha i}$
  
* 根号：\sqrt     eg：$ \sqrt[n]{5} $ 
  
* 省略号：

| LaTeX 显示 | 语句                                                |
| ---------- | --------------------------------------------------- |
| $ \dots$   | \dots                                               |
| $\vdots$   | \vdots                                              |
| $ \cdots$  | \cdots  (与上上行相比，这个省略号在一行的上下中部 ) |
| $ \cdot$   | \cdot                                               |
| $\ddots$   | \ddots                                              |



## 运算符

*   $\pm$  \\pm

*   $\div$  \div

*    $\le$  \le    

* 求和： \sum_1^n    eg：  $\sum_1^n$

* 积分： \int_1^n       eg： $\int_1^n$

* 极限：  lim_{x \to \infty}       eg：$lim {x \to \infty}$

* 分数：  \frac{}{}       eg：$\frac{3}{8}$

* 矩阵与行列式：

  * 矩阵： \begin{matrix}...\end{matrix}   ， 使用 & 分隔同行元素， \\ 表示换行
    $$
    \begin{matrix}
    	1 & x & x^2\\
    	1 & y & y^2\\
    	1 & z & z^2\\
    	\end{matrix}
    $$

  * 行列式：  X=\left|\begin{matrix}...\end{matrix}\right|
    $$
    X=\left|
    	\begin{matrix}
    		x_{11} & x_{12} & \cdots & x_{1d}\\
    		x_{21} & x_{22} & \cdots & x_{2d}\\
    		\vdots & \vdots & \ddots & \vdots \\
    		x_{11} & x_{12} & \cdots & x_{1d}\\
    	\end{matrix}
    \right|
    $$

* 分隔符

    各种括号用 () [] { } \langle\rangle 等命令表示,注意花括号通常用来输入命令和环境的参数,所以在数学公式中它们前面要加 \。可以在上述分隔符前面加 \big \Big \bigg \Bigg 等命令来调整大小。 

* 箭头

    | LaTeX 显示        | 语句            | LaTeX 显示            | 语句                |
    | ----------------- | --------------- | --------------------- | ------------------- |
    | $\leftarrow$      | \leftarrow      | $\longleftarrow$      | \longleftarrow      |
    | $\rightarrow$     | \rightarrow     | $\longrightarrow$     | \longrightarrow     |
    | $\leftrightarrow$ | \leftrightarrow | $\longleftrightarrow$ | \longleftrightarrow |
    | $\Leftarrow$      | \Leftarrow      | $\Longleftarrow$      | \Longleftarrow      |
    | $\Rightarrow$     | \Rightarrow     | $\Longrightarrow$     | \Longrightarrow     |
    | $\Leftrightarrow$ | \Leftrightarrow | $\Longleftrightarrow$ | \Longleftrightarrow |

* 方程式：\begin{equation}...\end{equation}
    $$
    \begin{equation}
        E=mc^2
        \end{equation}
    $$

* 分段函数： f(n)=\begin{cases}...\end{cases}
    $$
    f(n)=
    	\begin{cases}
    		n/2, & \text{if $n$ is even}\\
    		3n+1,& \text{if $n$ is odd}
    	\end{cases}
    $$

* 方程组：  \left\{ \begin{array}{}...\end{array} \right.        {}为必要的格式，原因暂时未知
    $$
    \left\{
    	\begin{array}{}
    		a_1x+b_1y+c_1z=d_1\\
    		a_2x+b_2y+c_2z=d_2\\
    		a_3x+b_3y+c_3z=d_3
    	\end{array}
    \right.
    $$

## 常用公式

* 线性模型

$$
h(\theta) = \sum_{j = 0} ^n \theta_j x_j
$$

* 均方误差

$$
J(\theta) = \frac{1}{2m}\sum_{i = 0} ^m(y^i - h_\theta (x^i))^2
$$

* 批量梯度下降

$$
\frac{\partial J(\theta)}{\partial\theta_j}=-\frac1m\sum_{i=0}^m(y^i-h_\theta(x^i))x^i_j
$$

* 

$$
\begin{align}
\frac{\partial J(\theta)}{\partial\theta_j}
& = -\frac1m\sum_{i=0}^m(y^i-h_\theta(x^i)) \frac{\partial}{\partial\theta_j}(y^i-h_\theta(x^i)) \\
& = -\frac1m\sum_{i=0}^m(y^i-h_\theta(x^i)) \frac{\partial}{\partial\theta_j}(\sum_{j=0}^n\theta_jx_j^i-y^i) \\
& = -\frac1m\sum_{i=0}^m(y^i-h_\theta(x^i))x^i_j
\end{align}
$$

* 

$$
X=\left(
        \begin{matrix}
            x_{11} & x_{12} & \cdots & x_{1d}\\
            x_{21} & x_{22} & \cdots & x_{2d}\\
            \vdots & \vdots & \ddots & \vdots\\
            x_{m1} & x_{m2} & \cdots & x_{md}\\
        \end{matrix}
    \right)
    =\left(
         \begin{matrix}
                x_1^T \\
                x_2^T \\
                \vdots\\
                x_m^T \\
            \end{matrix}
    \right)
$$

## 之后添加

* $ \top$    \top

* 字符下标
  $$
  \max \limits_{a<x<b} \{f(x)\}
  $$
  
*  $\LaTeX$      \LaTeX

* 字符上标
  $$
  \overset{n}{\sum \limits_{i=1}}
  $$
  
* 连乘号 $\prod_{i=1}^{n}$ 

* 波浪号   \sim           $\sim$

* \mathop{} 可以将在行间环境中无法正常显示上下方关系的符号变为正常显示。


## 矩阵的不同实现方式

$$
\begin{gather*}
\begin{matrix}
0 & 1\\
1 & 0
\end{matrix}
~~
\begin{pmatrix}
0 & 1\\
1 & 0
\end{pmatrix}
~~
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
~~
\begin{Bmatrix}
0 & 1\\
1 & 0
\end{Bmatrix}
~~
\begin{vmatrix}
0 & 1\\
1 & 0
\end{vmatrix}
~~
\begin{Vmatrix}
0 & 1\\
1 & 0
\end{Vmatrix}
\end{gather*}
$$

$$
\left(
\begin{matrix}
0 & 1\\
1 & 0
\end{matrix}
\right)
~~
\left|
\begin{matrix}
0 & 1\\
1 & 0
\end{matrix}
\right|
~~
\left[
\begin{matrix}
0 & 1\\
1 & 0
\end{matrix}
\right]
$$

## 一份文档

$The~more~unusual~symbols~are~not~defined~in~base~\LaTeX ~(NFSS)~and~require~\backslash usepackage\{amssymb\}$

### 1.0 $\LaTeX ~~ math ~~ constructs$  

|     LaTeX显示     | 语句            |     LaTeX显示     | 语句            |       LaTeX显示        | 语句                 |
| :---------------: | :-------------- | :---------------: | :-------------- | :--------------------: | :------------------- |
| $\frac{abc}{xyz}$ | \frac{abc}{xyz} | $\overline{abc}$  | \overline{abc}  | $\overrightarrow{abc}$ | \overrightarrow{abc} |
|       $f'$        | f'              | $\underline{abc}$ | \underline{abc} | $\overleftarrow{abc}$  | \overleftarrow{abc}  |
|   $\sqrt{abc}$    | \sqrt{abc}      |  $\widehat{abc}$  | \widehat{abc}   |   $\overbrace{abc}$    | \overbrace{abc}      |
|  $\sqrt[n]{abc}$  | \sqrt[n]{abc}   | $\widetilde{abc}$ | \widetilde{abc} |   $\underbrace{abc}$   | \underbrace{abc}     |

### 1.1 $Delimiters$

| LaTeX显示 | 语句 | LaTeX显示 | 语句    | LaTeX显示 | 语句    |  LaTeX显示   | 语句       |
| :-------: | :--- | :-------: | :------ | :-------: | :------ | :----------: | :--------- |
|    $|$    | \|   |  $\vert$  | \vert   |   $\|$    | \\| $\Vert$ |   $\Vert$    |
| $\{$ | \\{ | $\}$ | \\} | $\langle$ | \langle | $\rangle$ | \rangle |
| $\lfloor$ | \lfloor | $\rfloor$ | \rfloor | $\lceil$ | \lceil | $\rceil$ | \rceil |
| $/$ | / | $\backslash$ | \backslash | $[$ | [ | $]$ | ] |
| $\Uparrow$ | \Uparrow | $\uparrow$ | \uparrow | $\Downarrow$ | \Downarrow | $\downarrow$ | \downarrow |
| $\llcorner$ | \llcorner | $\lrcorner$ | \lrcorner | $\ulcorner$ | \ulcorner | $\urcorner$ | \urcorner |

$Use~the~pair~\backslash~left~s_1~and~\backslash~right~s_2~to~match~height~of~delimiters~s_1~and~s_2~to~the~height~of~their~contents~,~e.g.$

`\left|expr\right| \\\left\{expr\right\} \\\left\Vertexpr\right.`
$$
\left| expr \right| \\
\left\{ expr \right\} \\
\left\Vert expr \right.
$$

### 1.2 $Variable-sized~symbols~(displayed~formulae~show~larger~version)$

|  LaTeX显示  | 语句      |  LaTeX显示   | 语句       |  LaTeX显示  | 语句      |
| :---------: | --------- | :----------: | ---------- | :---------: | --------- |
|   $\sum$    | \sum      |   $\prod$    | \prod      |  $\coprod$  | \coprod   |
|   $\int$    | \int      |   $\oint$    | \oint      |   $\iint$   | \iint     |
| $\biguplus$ | \biguplus |  $\bigcap$   | \bigcap    |  $\bigcup$  | \bigcup   |
| $\bigoplus$ | \bigoplus | $\bigotimes$ | \bigotimes | $\bigodot$  | \bigodot  |
|  $\bigvee$  | \bigvee   | $\bigwedge$  | \bigwedge  | $\bigsqcup$ | \bigsqcup |

### 1.3 $Standard~Function~Names$

$Function~names~should~appear~in~Roman~,~not~Italic~,~e.g.$ 
$$
Correct:~\backslash tan(at-n\backslash pi) \rightarrow \tan(at-n\pi) \\
Incorrect:~tan(at-n\backslash pi) \rightarrow tan(at-n\pi)
$$

| LaTeX显示 | 语句    | LaTeX显示 | 语句    | LaTeX显示 | 语句    | LaTeX显示 | 语句    |
| :-------: | ------- | :-------: | ------- | :-------: | ------- | :-------: | ------- |
| $\arccos$ | \arccos | $\arcsin$ | \arcsin | $\arctan$ | \arctan |  $\arg$   | \arg    |
|  $\cos$   | \cos    |  $\cosh$  | \cosh   |  $\cot$   | \cot    |  $\coth$  | \coth   |
|  $\csc$   | \csc    |  $\deg$   | \deg    |   $det$   | \det    |  $\dim$   | \dim    |
|  $\exp$   | \exp    |  $\gcd$   | \gcd    |  $\hom$   | \hom    |  $\inf$   | \inf    |
|  $\ker$   | \ker    |   $\lg$   | \lg     |  $\lim$   | \lim    | $\liminf$ | \liminf |
| $\limsup$ | \limsup |   $\ln$   | \ln     |  $\log$   | \log    |  $\max$   | \max    |
|  $\min$   | \min    |   $\Pr$   | \Pr     |  $\sec$   | \sec    |  $\sin$   | \sin    |
|  $\sinh$  | \sinh   |  $\sup$   | \sup    |  $\tan$   | \tan    |  $\tanh$  | \tanh   |

### 1.4 $Binary~Operation/Relation~Symbols$

|      LaTeX显示      | 语句              |      LaTeX显示       | 语句               |       LaTeX显示       | 语句                |     LaTeX显示      | 语句             |
| :-----------------: | ----------------- | :------------------: | ------------------ | :-------------------: | ------------------- | :----------------: | ---------------- |
|       $\ast$        | \ast              |        $\pm$         | \pm                |        $\cap$         | \cap                |       $\lhd$       | \lhd             |
|       $\star$       | \star             |        $\mp$         | \mp                |        $\cup$         | \cup                |       $\rhd$       | \rhd             |
|       $\cdot$       | \cdot             |       $\amalg$       | \amalg             |       $\uplus$        | \uplus              |  $\triangleleft$   | \triangleleft    |
|       $\circ$       | \circ             |       $\odot$        | \odot              |       $\sqcap$        | \sqcap              |  $\triangleright$  | \triangleright   |
|  $\bull或\bullet$   | \bull或\bullet    |      $\ominus$       | \ominus            |       $\sqcup$        | \sqcup              |      $\unlhd$      | \unlhd           |
|     $\bigcirc$      | \bigcirc          |       $\oplus$       | \oplus             |       $\wedge$        | \wedge              |      $\unrhd$      | \unrhd           |
|     $\diamond$      | \diamond          |      $\oslash$       | \oslash            |        $\vee$         | \vee                | $\bigtriangledown$ | \bigtriangledown |
|      $\times$       | \times            |      $\otimes$       | \otimes            |       $\dagger$       | \dagger             |  $\bigtriangleup$  | \bigtriangleup   |
|       $\div$        | \div              |        $\wr$         | \wr                |      $\ddagger$       | \ddagger            |    $\setminus$     | \setminus        |
|    $\centerdot$     | \centerdot        |        $\Box$        | \Box               |      $\barwedge$      | \barwedge           |     $\veebar$      | \veebar          |
|    $\circledast$    | \circledast       |      $\boxplus$      | \boxplus           |     $\curlywedge$     | \curlywedge         |    $\curlyvee$     | \curlyvee        |
|   $\circledcirc$    | \circledcirc      |     $\boxminus$      | \boxminus          |        $\Cap$         | \Cap                |       $\Cup$       | \Cup             |
|   $\circleddash$    | \circleddash      |     $\boxtimes$      | \boxtimes          |        $\bot$         | \bot                |       $\top$       | \top             |
|     $\dotplus$      | \dotplus          |      $\boxdot$       | \boxdot            |      $\intercal$      | \intercal           | $\rightthreetimes$ | \rightthreetimes |
|  $\divideontimes$   | \divideontimes    |      $\square$       | \square            |   $\doublebarwedge$   | \doublebarwedge     | $\leftthreetimes$  | \leftthreetimes  |
|      $\equiv$       | \equiv            |        $\leq$        | \leq               |        $\geq$         | \geq                |      $\perp$       | \perp            |
|       $\cong$       | \cong             |       $\prec$        | \prec              |        $\succ$        | \succ               |       $\mid$       | \mid             |
|       $\neq$        | \neq              |      $\preceq$       | \preceq            |       $\succeq$       | \succeq             |    $\parallel$     | \parallel        |
|       $\sim$        | \sim              |        $\ll$         | \ll                |         $\gg$         | \gg                 |     $\bowtie$      | \bowtie          |
|      $\simeq$       | \simeq            |      $\subset$       | \subset            |       $\supset$       | \supset             |      $\Join$       | \Join            |
|      $\approx$      | \approx           |     $\subseteq$      | \subseteq          |      $\supseteq$      | \supseteq           |     $\ltimes$      | \ltimes          |
|      $\asymp$       | \asymp            |     $\sqsubset$      | \sqsubset          |      $\sqsupset$      | \sqsupset           |     $\rtimes$      | \rtimes          |
|      $\doteq$       | \doteq            |    $\sqsubseteq$     | \sqsubseteq        |     $\sqsupseteq$     | \sqsupseteq         |      $\smile$      | \smile           |
|      $\propto$      | \propto           |       $\dashv$       | \dashv             |       $\vdash$        | \vdash              |      $\frown$      | \frown           |
|      $\models$      | \models           |        $\in$         | \in                |         $\ni$         | \ni                 |      $\notin$      | \notin           |
|     $\approxeq$     | \approxeq         |       $\leqq$        | \leqq              |        $\geqq$        | \geqq               |     $\lessgtr$     | \lessgtr         |
|     $\thicksim$     | \thicksim         |     $\leqslant$      | \leqslant          |      $\geqslant$      | \geqslant           |    $\lesseqgtr$    | \lesseqgtr       |
|     $\backsim$      | \backsim          |    $\lessapprox$     | \lessapprox        |     $\gtrapprox$      | \gtrapprox          |   $\lesseqqgtr$    | \lesseqqgtr      |
|    $\backsimeq$     | \backsimeq        |        $\lll$        | \lll               |        $\ggg$         | \ggg                |   $\gtreqqless$    | \gtreqqless      |
|    $\triangleq$     | \triangleq        |      $\lessdot$      | \lessdot           |       $\gtrdot$       | \gtrdot             |    $\gtreqless$    | \gtreqless       |
|      $\circeq$      | \circeq           |      $\lesssim$      | \lesssim           |       $\gtrsim$       | \gtrsim             |     $\gtrless$     | \gtrless         |
|      $\bumpeq$      | \bumpeq           |    $\eqslantless$    | \eqslantless       |     $\eqslantgtr$     | \eqslantgtr         |   $\backepsilon$   | \backepsilon     |
|      $\Bumpeq$      | \Bumpeq           |      $\precsim$      | \precsim           |      $\succsim$       | \succsim            |     $\between$     | \between         |
|     $\doteqdot$     | \doteqdot         |    $\precapprox$     | \precapprox        |     $\succapprox$     | \succapprox         |    $\pitchfork$    | \pitchfork       |
|   $\thickapprox$    | \thickapprox      |      $\Subset$       | \Subset            |       $\Supset$       | \Supset             |    $\shortmid$     | \shortmid        |
|  $\fallingdotseq$   | \fallingdotseq    |     $\subseteqq$     | \subseteqq         |     $\supseteqq$      | \supseteqq          |   $\smallfrown$    | \smallfrown      |
|   $\risingdotseq$   | \risingdotseq     |     $\sqsubset$      | \sqsubset          |      $\sqsupset$      | \sqsupset           |   $\smallsmile$    | \smallsmile      |
|    $\varpropto$     | \varpropto        |    $\preccurlyeq$    | \preccurlyeq       |    $\succcurlyeq$     | \succcurlyeq        |      $\Vdash$      | \Vdash           |
|    $\therefore$     | \therefore        |    $\curlyeqprec$    | \curlyeqprec       |    $\curlyeqsucc$     | \curlyeqsucc        |      $\vDash$      | \vDash           |
|     $\because$      | \because          | $\blacktriangleleft$ | \blacktriangleleft | $\blacktriangleright$ | \blacktriangleright |     $\Vvdash$      | \Vvdash          |
|      $\eqcirc$      | \eqcirc           |  $\trianglelefteq$   | \trianglelefteq    |  $\trianglerighteq$   | \trianglerighteq    |  $\shortparallel$  | \shortparallel   |
|                     |                   |  $\vartriangleleft$  | \vartriangleleft   |  $\vartriangleright$  | \vartriangleright   | $\nshortparallel$  | \nshortparallel  |
|      $\ncong$       | \ncong            |       $\nleq$        | \nleq              |        $\ngeq$        | \ngeq               |    $\nsubseteq$    | \nsubseteq       |
|       $\nmid$       | \nmid             |       $\nleqq$       | \nleqq             |       $\ngeqq$        | \ngeqq              |    $\nsupseteq$    | \nsupseteq       |
|    $\nparallel$     | \nparallel        |     $\nleqslant$     | \nleqslant         |     $\ngeqslant$      | \ngeqslant          |   $\nsubseteqq$    | \nsubseteqq      |
|    $\nshortmid$     | \nshortmid        |       $\nless$       | \nless             |        $\ngtr$        | \ngtr               |   $\nsupseteqq$    | \nsupseteqq      |
|  $\nshortparallel$  | \nshortparallel   |       $\nprec$       | \nprec             |       $\nsucc$        | \nsucc              |    $\subsetneq$    | \subsetneq       |
|       $\nsim$       | \nsim             |      $\npreceq$      | \npreceq           |      $\nsucceq$       | \nsucceq            |    $\supsetneq$    | \supsetneq       |
|      $\nVDash$      | \nVDash           |    $\precnapprox$    | \precnapprox       |    $\succnapprox$     | \succnapprox        |   $\subsetneqq$    | \subsetneqq      |
|      $\nvDash$      | \nvDash           |     $\precnsim$      | \precnsim          |      $\succnsim$      | \succnsim           |   $\supsetneqq$    | \supsetneqq      |
|      $\nvdash$      | \nvdash           |     $\lnapprox$      | \lnapprox          |      $\gnapprox$      | \gnapprox           |  $\varsubsetneq$   | \varsubsetneq    |
|  $\ntriangleleft$   | \ntriangleleft    |       $\lneq$        | \lneq              |        $\gneq$        | \gneq               |  $\varsupsetneq$   | \varsupsetneq    |
| $\ntrianglelefteq$  | \ntrianglelefteq  |       $\lneqq$       | \lneqq             |       $\gneqq$        | \gneqq              |  $\varsubsetneqq$  | \varsubsetneqq   |
|  $\ntriangleright$  | \ntriangleright   |       $\lnsim$       | \lnsim             |       $\gnsim$        | \gnsim              |  $\varsupsetneqq$  | \varsupsetneqq   |
| $\ntrianglerighteq$ | \ntrianglerighteq |     $\lvertneqq$     | \lvertneqq         |     $\gvertneqq$      | \gvertneqq          |                    |                  |

### 1.5 $Arrow~symbols$

|      LaTeX显示       | 语句               |       LaTeX显示        | 语句                 |      LaTeX显示       | 语句               |
| :------------------: | ------------------ | :--------------------: | -------------------- | :------------------: | ------------------ |
|     $\leftarrow$     | \leftarrow         |      $\leftarrow$      | \leftarrow           |      $\uparrow$      | \uparrow           |
|     $\Leftarrow$     | \Leftarrow         |    $\Longleftarrow$    | \Longleftarrow       |      $\Uparrow$      | \Uparrow           |
|    $\rightarrow$     | \rightarrow        |   $\longrightarrow$    | \longrightarrow      |     $\downarrow$     | \downarrow         |
|    $\Rightarrow$     | \Rightarrow        |   $\Longrightarrow$    | \Longrightarrow      |     $\Downarrow$     | \Downarrow         |
|  $\leftrightarrow$   | \leftrightarrow    | $\longleftrightarrow$  | \longleftrightarrow  |    $\updownarrow$    | \updownarrow       |
|  $\Leftrightarrow$   | \Leftrightarrow    | $ \Longleftrightarrow$ | \Longleftrightarrow  |    $\Updownarrow$    | \Updownarrow       |
|      $\mapsto$       | \mapsto            |     $\longmapsto$      | \longmapsto          |      $\nearrow$      | \nearrow           |
|   $\hookleftarrow$   | \hookleftarrow     |   $\hookrightarrow$    | \hookrightarrow      |      $\searrow$      | \searrow           |
|   $\leftharpoonup$   | \leftharpoonup     |   $\rightharpoonup$    | \rightharpoonup      |      $\swarrow$      | \swarrow           |
|  $\leftharpoondown$  | \leftharpoondown   |  $\rightharpoondown$   | \rightharpoondown    |      $\nwarrow$      | \nwarrow           |
| $\rightleftharpoons$ | \rightleftharpoons |       $\leadsto$       | \leadsto             |                      |                    |
|  $\dashrightarrow$   | \dashrightarrow    |    $\dashleftarrow$    | \dashleftarrow       |  $\leftleftarrows$   | \leftleftarrows    |
|  $\leftrightarrows$  | \leftrightarrows   |     $\Lleftarrow$      | \Lleftarrow          | $\twoheadleftarrow$  | \twoheadleftarrow  |
|   $\leftarrowtail$   | \leftarrowtail     |    $\looparrowleft$    | \looparrowleft       | $\leftrightharpoons$ | \leftrightharpoons |
|  $\curvearrowleft$   | \curvearrowleft    |   $\circlearrowleft$   | \circlearrowleft     |       $ \Lsh$        | \Lsh               |
|    $\upuparrows$     | \upuparrows        |    $\upharpoonleft$    | \upharpoonleft       |  $\downharpoonleft$  | \downharpoonleft   |
|     $\multimap$      | \multimap          | $\leftrightsquigarrow$ | \leftrightsquigarrow |  $\rightleftarrows$  | \rightleftarrows   |
|  $\rightleftarrows$  | \rightleftarrows   |  $\rightrightarrows$   | \rightrightarrows    |                      |                    |
| $\twoheadrightarrow$ | \twoheadrightarrow |   $\rightarrowtail$    | \rightarrowtail      |  $\looparrowright$   | \looparrowright    |
| $\rightleftharpoons$ | \rightleftharpoons |   $\curvearrowright$   | \curvearrowright     | $\circlearrowright$  | \circlearrowright  |
|        $\Rsh$        | \Rsh               |   $\downdownarrows$    | \downdownarrows      |  $\upharpoonright$   | \upharpoonright    |
| $\downharpoonright$  | \downharpoonright  |   $\rightsquigarrow$   | \rightsquigarrow     |                      |                    |
|    $\nleftarrow$     | \nleftarrow        |     $\nrightarrow$     | \nrightarrow         |    $\nLeftarrow$     | \nLeftarrow        |
|    $\nRightarrow$    | \nRightarrow       |   $\nleftrightarrow$   | \nleftrightarrow     |  $\nLeftrightarrow$  | \nLeftrightarrow   |

### 1.6 $Miscellaneous~symbols$

|   LaTeX显示    | 语句         |   LeTaX显示   | 语句        |  LeTaX显示  | 语句      |        LaTeX显示         | 语句                               |
| :------------: | ------------ | :-----------: | ----------- | :---------: | --------- | :----------------------: | ---------------------------------- |
|    $\infty$    | \infty       |   $\forall$   | \forall     |   $\Bbbk$   | \Bbbk     |          $\wp$           | \wp                                |
|    $\nabla$    | \nabla       |   $\exists$   | \exists     | $\bigstar$  | \bigstar  |         $\angle$         | \angle                             |
|   $\partial$   | \partial     |  $\nexists$   | \nexists    | $\diagdown$ | \diagdown |     $\measuredangle$     | \measuredangle                     |
|     $\eth$     | \eth         |  $\emptyset$  | \emptyset   |  $\diagup$  | \diagup   |    $\sphericalangle$     | \sphericalangle                    |
|  $\clubsuit$   | \clubsuit    | $\varnothing$ | \varnothing | $\Diamond$  | \Diamond  |      $\complement$       | \complement                        |
| $\diamondsuit$ | \diamondsuit |   $\imath$    | \imath      |   $\Finv$   | \Finv     |     $\triangledown$      | \triangledown                      |
|  $\heartsuit$  | \heartsuit   |   $\jmath$    | \jmath      |   $\Game$   | \Game     |       $\triangle$        | \triangle                          |
|  $\spadesuit$  | \spadesuit   |    $\ell$     | \ell        |   $\hbar$   | \hbar     |      $\vartriangle$      | \vartriangle                       |
|    $\cdots$    | \cdots       |   $\iiiint$   | \iiiint     |  $\hslash$  | \hslash   |     $\blacklozenge$      | \blacklozenge                      |
|    $\vdots$    | \vdots       |   $\iiint$    | \iiint      | $\lozenge$  | \lozenge  |      $\blacksquare$      | \blacksquare                       |
|    $\ldots$    | \ldots       |    $\iint$    | \iint       |   $\mho$    | \mho      |     $\blacktriangle$     | \blacktriangle                     |
|    $\ddots$    | \ddots       |   $\sharp$    | \sharp      |  $\prime$   | \prime    |   $\blacktriangledown$   | \blacktriangledown                 |
|     $\Im$      | \Im          |    $\flat$    | \flat       |  $\square$  | \square   |       $\backprime$       | \backprime                         |
|     $\Re$      | \Re          |  $\natural$   | \natural    |   $\surd$   | \surd     | $\circledR and\circledS$ | \circledR <br />and<br />\circledS |

### 1.7 $Math~mode~accents$

|  LaTeX显示  | 语句      |  LaTeX显示  | 语句      |
| :---------: | --------- | :---------: | --------- |
| $\acute{a}$ | \acute{a} |  $\bar{a}$  | \bar{a}   |
| $\breve{a}$ | \breve{a} | $\check{a}$ | \check{a} |
| $\ddot{a}$  | \ddot{a}  |  $\dot{a}$  | \dot{a}   |
| $\grave{a}$ | \grave{a} |  $\hat{a}$  | \hat{a}   |
| $\tilde{a}$ | \tilde{a} |  $\vec{a}$  | \vec{a}   |

### 1.8 $Array~environment~,~examples$

Simplest version:

```latex
\begin{array}{cols} 
row_1 & 1 \\ 
row_2 & 2\\
\dots & \dots \\
row_m & m
\end{array}
```


$$
\begin{array}{cols} 
row_1 & 1 \\ 
row_2 & 2\\
\dots & \dots \\
row_m & m
\end{array}
$$

```latex
\left(
\begin{array}{cc} 
2\tau & 7\phi-\frac5{12} \\
3\psi & \frac{\pi}8 
\end{array}
\right )
\left(
\begin{array}{c}
x\\y
\end{array}
\right )
~\mbox{and}~
\left[
\begin{array}{cc|r}
3 & 4 & 5 \\
1 & 3 & 729
\end{array}
\right]
```

$$
\left(
\begin{array}{cc} 
2\tau & 7\phi-\frac5{12} \\
3\psi & \frac{\pi}8 
\end{array}
\right )
\left(
\begin{array}{c}
x\\y
\end{array}
\right )
~\mbox{and}~
\left[
\begin{array}{cc|r}
3 & 4 & 5 \\
1 & 3 & 729
\end{array}
\right]
$$

```latex
f(z) = 
\left\{
\begin{array}{rcl}
\overline{\overline{z^2}+\cos z} & \mbox{for} & |z|<3 \\
0 & \mbox{for} & 3\leq|z|\leq5 \\
\sin\overline{z} & \mbox{for} & |z|>5
\end{array}
\right.
```

$$
f(z) = 
\left\{
\begin{array}{rcl}
\overline{\overline{z^2}+\cos z} & \mbox{for} & |z|<3 \\
0 & \mbox{for} & 3\leq|z|\leq5 \\
\sin\overline{z} & \mbox{for} & |z|>5
\end{array}
\right.
$$

### 1.9 $Other~Styles~(math~mode~only)$

Caligraphic letters: `$\mathcal{A}$` etc.: $\mathcal{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$

Mathbb letters: `$\mathbb{A}$` etc.: $\mathbb{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$

Mathfrak letters: `$\mathfrak{A}$` etc.: $\mathfrak{ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdef123456}$

Math Sans serif letters: `$\mathsf{A}$` etc.: $\mathsf{ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdef 123456}$

Math bold letters: `$\mathbf{A}$` etc.: $\mathbf{ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdef 123456}$

花体字母：`$\mathscr{A}$` etc.: $\mathscr{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$ 

### 1.10 $Font~sizes$

#### 1.10.0 $Math~Mode$

| LaTeX显示                                     | 语句                                          |
| --------------------------------------------- | --------------------------------------------- |
| ${\displaystyle \int f^{-1}(x-x_a)\,dx}$      | `{\displaystyle \int f^{-1}(x-x_a)\,dx}`      |
| ${\textstyle \int f^{-1}(x-x_a)\,dx}$         | `{\textstyle \int f^{-1}(x-x_a)\,dx}`         |
| ${\scriptstyle \int f^{-1}(x-x_a)\,dx}$       | `{\scriptstyle \int f^{-1}(x-x_a)\,dx}`       |
| ${\scriptscriptstyle \int f^{-1}(x-x_a)\,dx}$ | `{\scriptscriptstyle \int f^{-1}(x-x_a)\,dx}` |

#### 1.10.1 $Text~Mode$

|                LaTeX显示 | 语句                      |
| -----------------------: | ------------------------- |
|         $\tiny smallest$ | `\tiny smallest`          |
| $\scriptsize very small$ | `\scriptsize very small$` |
|           $\small small$ | `\small small$`           |
|     $\normalsize normal$ | `\normalsize normal`      |
|           $\large large$ | `\large large`            |
|           $\Large Large$ | `\Large Large`            |
|           $\LARGE LARGE$ | `\LARGE LARGE`            |
|             $\huge huge$ | `\huge huge`              |
|             $\Huge Huge$ | `\Huge Huge`              |

### 1.11 $Text~Mode:~Accents~and~Symbols$

| LaTeX显示 | 语句      | LaTeX显示 | 语句 | LaTeX显示 | 语句  |
| :-------: | --------- | :-------: | ---- | :-------: | ----- |
|   $\P$    | `\P`      |   $\S$    | `\S` |   $\AA$   | `\AA` |
| $\dagger$ | `\dagger` |           |      |           |       |



