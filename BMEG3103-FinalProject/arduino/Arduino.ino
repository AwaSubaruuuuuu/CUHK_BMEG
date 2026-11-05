/*
 通过UART串口显示信号值、注意力及放松度的值
 */
#define BAUDRATE 57600
#define DEBUGOUTPUT 0

//校验相关变量
int   generatedChecksum = 0;
byte  checksum = 0; 

//接收数据长度和数据数组
byte  payloadLength = 0;
byte  payloadData[32] = {0};//总共接收32个自己的数据

//需要读取的信息变量
byte signalquality = 0;//信号质量
byte attention = 0;    //注意力值
byte meditation = 0;   //放松度值

long feature[9] = {0};
int feature_len = 9;

//初始化
void setup() 
{
    Serial.begin(BAUDRATE);
    Serial.setTimeout(1);  // Decrease Serial timeout
}

//从串口读取一个字节数据
byte ReadOneByte() 
{
  int ByteRead;
  while(!Serial.available());
  ByteRead = Serial.read();
  return ByteRead;//返回读到的字节
}

void passFeature(long feature[], int len)
{
    // Send all features in one burst
    for(int i = 0; i < len; i++)
    {
        Serial.println(feature[i]);  // Using println for compatibility
    }
    Serial.flush();  // Ensure all data is sent
}


//读取串口数据
void read_serial_data()
{
    //寻找数据包起始同步字节，2个
    if(ReadOneByte() == 0xAA)//先读一个
    {
      if(ReadOneByte() == 0xAA)//读第二个
      {
        payloadLength = ReadOneByte();//读取第三个，数据包字节的长度
        if(payloadLength == 0x20)//如果接收到的是大包数据才继续读取，小包数据则舍弃不读取
        {
          generatedChecksum = 0; //校验变量清0       
          for(int i = 0; i < payloadLength; i++)//连续读取32个字节
          {  
            payloadData[i] = ReadOneByte();//读取指定长度数据包中的数据
            generatedChecksum += payloadData[i];//计算数据累加和
          }         
          checksum = ReadOneByte();//读取校验字节  
          //校验
          generatedChecksum = (~generatedChecksum)&0xff;       
          //比较校验字节
          if(checksum == generatedChecksum)//数据接收正确，继续处理 
          {    
            if(payloadData[1] == 0)
            {
            int mod = 0;
            int order = 0;
            long tmp = 0;
            #if !DEBUGOUTPUT         
            for(int j = 4; j < 28; j++)
            {
              order = (j - 6) / 3;
              mod = (j - 3) % 3 ;
              switch (mod) 
              {
                case 1:

                tmp = payloadData[j] + tmp;
               


                break ;
                case 2:

                tmp = payloadData[j] * 256 + tmp;

              

                break; 
                case 0:

                tmp = (payloadData[j] * 65536) + tmp;
                
                
                feature[order] = tmp ;
                tmp = 0;
                

                break;
              }
            }  
            feature[8] = payloadData[29];
            passFeature(feature, feature_len);
            
            #endif  
            
            }
          } 
        } 
      }
    }
}

//主循环
void loop() 
{
  read_serial_data();//读取串口数据 
}
