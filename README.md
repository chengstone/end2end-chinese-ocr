# end2end-chinese-ocr
This is end2end chinese ocr demo using CNN + BiLSTM + ctc_loss

这是一个端对端中文OCR的Demo程序，只使用了一种字体进行训练，使用的是一级汉字字符集。

本Demo程序只为了验证技术有效性，如果您想用于自己的项目中，
可以加入更多的字体（在代码中找font_list），并且扩展字符集（在代码中找charset）重新训练，比如加入二级汉字、数字、字母和标点符号等等。

关于汉字字符集您可以参考这个网址：https://www.qqxiuzi.cn/zh/yijiziku-erjiziku.php

为了得到更好的识别效果，建议您加入适当的数据增强（因为相关代码我已经从代码中删除了），包括适当缩小汉字的尺寸，以适应分辨率低的文字图像。

关于数据增强的入口，我已经在代码中预留了，您只要在imgaug_process函数中加入相应实现即可。

另外，代码中实现了5种LSTM，通过参数birnn_type进行选择：

0:dynamic_rnn，普通的两层RNN网络。

1:static_bidirectional_rnn，静态非栈式双向LSTM。前向LSTM和后向LSTM的层数由参数self.lstm_num_layers指定，默认是1层。

静态的含义是输入数据的timestep是固定的，也就是说每张汉字图片的宽度是固定的。

需要传给参数max_timestep一个固定值，这个值就是输入图片经过CNN最后一层后的宽度，公式是(宽度 // 16)。

因为CNN一共四层，经过4个max pool之后，相当于宽度缩小了16倍（2的4次幂）。CNN的层数由参数self.cnn_layer_num指定。

2:bidirectional_dynamic_rnn，动态非栈式双向LSTM。前向LSTM和后向LSTM的层数由参数self.lstm_num_layers指定，默认是1层。

动态的含义是输入数据的timestep不固定，输入数据可以接受不同宽度的汉字图片。为了简化代码，每个批次（batch）的输入数据宽度是一致的。

如果您非要使用每张图片的宽度都不一样，请修改self.seq_len。

3:stack_bidirectional_dynamic_rnn，动态栈式双向LSTM。除了LSTM的结构跟非栈式不一样之外，其他不变。

关于栈式和非栈式的区别，您可以参考：https://stackoverflow.com/questions/49242266/difference-between-multirnncell-and-stack-bidirectional-dynamic-rnn-in-tensorflo

4:stack_bidirectional_rnn，静态栈式双向LSTM。

虽然实现了5种，但其实影响网络结构的只有三种，普通RNN网络、栈式LSTM和非栈式LSTM。静态和动态的区别只跟输入数据的尺寸有关。

本项目提供的模型文件是2:bidirectional_dynamic_rnn，动态非栈式双向LSTM训练的结果，当然只训练了10k次，训练并不充分。您也可以尝试使用栈式的结构训练。

网络的结构有很多地方可以微调，您都可以修改试试。

比如CNN的结构、CNN层数、filter size、卷积核的参数、隐节点的个数、输入数据的尺寸、批次大小、LSTM的种类、LSTM的层数等。

运行命令python ocr.py进行训练

识别图片：
![image](https://github.com/chengstone/end2end-chinese-ocr/raw/master/newimg.png)

识别结果：

(1, 32, ?, 32)

(1, 16, ?, 64)

(1, 8, ?, 128)

(1, 4, ?, 128)

lstm input shape: [1, None, 512]

Successfully loaded: ./models/best_model.ckpt-10160

-------- prediction : ['滚', '滚', '长', '江', '东', '逝', '水', '浪', '花', '淘', '尽', '英', '雄'] --------
