from config import LOG_INTERVAL

# 训练过程
def train(model, loss_fn, device, train_loader, optimizer, epoch):
    # 开启梯度计算
    model.train()
    for batch_idx, (data_input, true_label) in enumerate(train_loader):
        # 从数据加载器读取一个batch
        # 把数据上载到GPU（如有）
        data_input, true_label = data_input.to(device), true_label.to(device)
        # 求解器初始化（每个batch初始化一次）
        optimizer.zero_grad()
        # 正向传播：模型由输入预测输出
        output = model(data_input)
        # 计算loss
        loss = loss_fn(output, true_label)
        # 反向传播：计算当前batch的loss的梯度
        loss.backward()
        # 由求解器根据梯度更新模型参数
        optimizer.step()

        # 间隔性的输出当前batch的训练loss
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
