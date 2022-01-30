训练参数: D:\opt\conda\envs\dl\python.exe D:/workspace/CATL-line/main.py --n_iters=100000 --device=cuda --base_dir=./datasets/images --indices=2,3,4,5,6,10 --batch_size=1 --num_workers=1 --optimizer=Adam --loss=MSE --lr=0.00005 --info_interval=100

输入：虚2.bmp,虚3.bmp,虚4.bmp,虚5.bmp,虚6.bmp,虚10.bmp,
模拟人工画bounding-box，随机截取包含两条竖线（标签）的长方形区域，
resize到256x16，标签归一化到宽度占比0~1浮点，输入网络

网络模型：models.net.Net

优化器：Adam，学习率：0.00005，loss：MSE

输出：x1,x2表示两条竖线（标签）x轴位置百分比0~1

抽样日志输出格式（转换百分比到x轴像素点坐标）：
label: (lx1, lx2) pred: (px1, px2) delta(pixel): abs(lx1-px1)+abs(lx2-px2)
