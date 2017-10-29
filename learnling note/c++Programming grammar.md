10/2/2017 10:58:35 AM 
# c++内联函数 #
## 背景（内联函数相对一般函数和宏代码的优点） ##
- 调用函数比求解等价表达式要慢得多。在大多数的机器上，调用函数都要做很多工作：调用前要先保存寄存器，并在返回时恢复，复制实参，程序还必须转向一个新位置执行
- 内联函数在编译时将函数展开为表达式从而消除了把 max写成函数的额外执行开销
- 使用宏代码最大的缺点是容易出错，预处理器在拷贝宏代码时常常产生意想不到的边际效应
- 宏代码不可调试，内联函数可以调试
- 宏代码无法操作类的私有成员
- 内联函数被内联后，编译器就看他通过上下文相关的技术对结果代码执行更深入的优化

## 注意事项 ##
-  关键字 inline 必须与函数定义体放在一起才能使函数成为内联，仅将 inline 放在函数声明前面不起任何作用
-  内联函数应该在头文件中定义
-  只有当函数只有 10 行甚至更少时才将其定义为内联函数
-  当函数被声明为内联函数之后, 编译器会将其内联展开, 而不是按通常的函数调用机制进行调用.
-  当函数体比较小的时候, 内联该函数可以令目标代码更加高效. 对于存取函数以及其它函数体比较短, 性能关键的函数, 鼓励使用内联.
-  滥用内联将导致程序变慢. 内联可能使目标代码量或增或减, 这取决于内联函数的大小. 内联非常短小的存取函数通常会减少代码大小, 但内联一个相当大的函数将戏剧性的增加代码大小. 现代处理器由于更好的利用了指令缓存, 小巧的代码往往执行更快。
-  一个较为合理的经验准则是, 不要内联超过 10 行的函数. 谨慎对待析构函数, 析构函数往往比其表面看起来要更长, 因为有隐含的成员和基类析构函数被调用!
-  内联那些包含循环或 switch 语句的函数常常是得不偿失 (除非在大多数情况下, 这些循环或 switch 语句从不被执行).
-  虚函数和递归函数就不会被正常内联.