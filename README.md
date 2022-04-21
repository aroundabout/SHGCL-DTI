# CoDTI

同济大学软件学院2022年毕业设计

1850231 姚凯楠



项目说明

1. 直接 run src/application/main.py即可 tools.tools.args parse_argsCO里面指定device='cuda:3'

1. data数据
   1. data dti相关的交互矩阵 包括drug protein sideeffect disease
   2. feature 生成的drug protein sideeffect disease的特征
   3. mp 元路径
   4. pos 对比学习正样本
2. src源码
   1. application
      1. main.py 直接运行
   2. data_process
      1. GetMp.py 生成元路径
      2. GetPos.py 生成正样本
      3. 别的都是生成节点特征用的
   3. layers
      1. mp_encoder.py HeCo的元路径视角编码器
      2. sc_encoder.py HeCo的邻居视角的编码器
      3. constrast.py HeCo的对比层
      4. MLPPredicator.py 三层的MLP 里面的两个MLP模型是一样的 MLPPredicatorDTI.py 方便使用一点
   4. model
      1. CoGnnNet.py HeCo的实现
   5. tools
      1. args EarlyStopping
      2. tools 写代码过程中各种方法都往里面放了

