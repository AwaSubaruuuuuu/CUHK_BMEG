/*
  TGAM / MindWave 大包解析与转发
  ---------------------------------------------------------------
  每收到一个 38 字节大包，在串口打印 9 个逗号分隔的字段：

    signal,attention,meditation,delta,theta,low_alpha,high_alpha,low_beta,high_beta

  字节布局来自对 tools/UartAssist.cfg 里三帧真实抓包的逐字节核对：
    [0] raw 低      [1] raw 高      [2] 保留(0x83)
    [3] 信号质量    [4..27] 8 段功率，每段 3 字节大端
    [28] 保留(0x04) [29] 注意力     [30] 放松度   [31] 保留

  注意：Arduino 的串口和 TGAM 模块共用了硬件串口，所以这里的
  Serial 既是"读 TGAM"又是"写电脑"。重烧固件前必须拔掉 TGAM 的 TXD，
  否则上传会被脑波数据打断（avrdude: stk500_recv(): programmer is not responding）。
  更稳妥的做法是把 TGAM 接到软串口，见 README。
 */

#define BAUDRATE 57600
#define DEBUGOUTPUT 0

// 校验相关变量
int  generatedChecksum = 0;
byte checksum = 0;

// 接收数据长度和数据数组
byte payloadLength = 0;
byte payloadData[32] = {0};

// 需要读取的信息变量
byte signalquality = 0;  // 信号质量 0-200，越大越差，0 表示接触良好
byte attention = 0;      // 注意力 0-100
byte meditation = 0;     // 放松度 0-100

// 8 段 eeg_power 累加和，和 TGAM 的 eegPower 总和对照可校验解析是否正确
unsigned long eegPowerSum = 0;

void setup()
{
    Serial.begin(BAUDRATE);
    Serial.setTimeout(1);           // 降低阻塞式读取的超时
    Serial.println("field,signal,attention,meditation,delta,theta,low_alpha,high_alpha,low_beta,high_beta");
}

// 从串口读取一个字节数据
byte ReadOneByte()
{
    while (!Serial.available());
    return (byte)Serial.read();
}

// 打印一行 9 个字段：signal,attention,meditation,8 段中前 6 段
void passFeature(byte signal, byte attn, byte medi, const unsigned long power[], int n)
{
    Serial.print(signal, DEC);
    Serial.print(',');
    Serial.print(attn, DEC);
    Serial.print(',');
    Serial.print(medi, DEC);
    for (int k = 0; k < n; k++)
    {
        Serial.print(',');
        Serial.print(power[k], DEC);
    }
    Serial.println();
    Serial.flush();                 // 阻塞到数据发完，避免下一个包覆盖
}

// 读取串口数据
void read_serial_data()
{
    // 寻找数据包起始同步字节，2 个 0xAA
    if (ReadOneByte() == 0xAA)
    {
        if (ReadOneByte() == 0xAA)
        {
            payloadLength = ReadOneByte();
            // 只处理大包(0x20)；小包(0x04)是单点原始波，这里丢弃
            if (payloadLength == 0x20)
            {
                generatedChecksum = 0;
                for (int i = 0; i < payloadLength; i++)
                {
                    payloadData[i] = ReadOneByte();
                    generatedChecksum += payloadData[i];
                }
                checksum = ReadOneByte();
                generatedChecksum = (~generatedChecksum) & 0xFF;
                if (checksum != generatedChecksum)      // 校验失败，丢弃
                    return;

                // ---- 关键偏移 ----
                signalquality = payloadData[3];         // [3] 才是信号质量
                attention     = payloadData[29];
                meditation    = payloadData[30];        // [30] 是放松度

#if !DEBUGOUTPUT
                // 8 段功率：从 [4] 开始，每段 3 字节大端
                unsigned long power[8];
                eegPowerSum = 0;
                for (int k = 0; k < 8; k++)
                {
                    int o = 4 + k * 3;
                    power[k] = ((unsigned long)payloadData[o] << 16)
                             | ((unsigned long)payloadData[o + 1] << 8)
                             | ((unsigned long)payloadData[o + 2]);
                    eegPowerSum += power[k];
                }
                // [28] 是保留位(实测恒为 0x04)，不是第 8 段功率；
                // 因此这里只转发前 6 段：delta/theta/low_alpha/high_alpha/low_beta/high_beta。
                // 让上位机额外转发到第 9 个字段，会破坏下游按固定列宽解析的假设。
                // 下游若确实需要 low_gamma/mid_gamma，把 n 改成 8 即可，但要先改上位机解析器。
                passFeature(signalquality, attention, meditation, power, 6);
#endif
            }
        }
    }
}

// 主循环
void loop()
{
    read_serial_data();
}
