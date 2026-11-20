/* =============================================================================
 * nfh_bridge.ino  ——  NeuroFocus 注意力热力图 · Arduino 透传桥 + 本地反馈
 *
 * 角色：把 TGAM（MindWave 头带里的脑电模块）的 UART 字节流**原样**转给电脑的
 *       USB 串口，同时由 Arduino 自己做三件电脑做不了/做不快的事：
 *
 *   1) 断链看门狗：TGAM 停止发帧 2.5 s 就在状态位里置 link=0，电脑侧立刻能看到
 *      「头带掉了 / 蓝牙断了」，不用等模型输出异常。
 *   2) 物理按键标记：按一下 D4 上的按钮，立刻往电脑推一个 MARK 帧（毫秒级），
 *      用来标记「我没看懂这里」「这里走神了」，比事后回忆时间点可靠。
 *   3) 片上 LED 反馈：attention 低于阈值时 LED 呼吸闪，高于阈值时随分数变亮。
 *      这个回路完全在片上，不经电脑、不经浏览器，即使串口拥塞也在闪。
 *
 * 另外它还负责把 TGAM 的 3.3V 电平和电脑的 USB 隔开，并给裸 TGAM 模块供电。
 *
 * ---------------------------------------------------------------------------
 * 接线（默认配置）
 * ---------------------------------------------------------------------------
 *   TGAM 板            Arduino
 *   ---------------------------------------------------------------
 *   TXD   ──────────►  D2   (TGAM→Arduino，只用这一根信号线)
 *   VCC   ──────────►  3.3V （裸 TGAM 模块必须 3.3V，不要接 5V！）
 *   GND   ──────────►  GND
 *   RXD   ──────✕ 不接（本固件从不向 TGAM 写任何东西）
 *
 *   按键：D4 ── 按钮 ── GND（用内部上拉）
 *   RGB/单色 LED：D9（PWM，串 220Ω 到 GND）
 *   板载心跳：D13（板载 LED，收到 TGAM 帧时闪）
 *
 * ---------------------------------------------------------------------------
 * 关于 Uno 的 57600 软串口
 * ---------------------------------------------------------------------------
 *   TGAM 的 raw 是 512 Hz（每帧 5 字节），加上 1 Hz 大包，总流量约 21 kbit/s。
 *   在 57600 波特率下占 37%，Uno 的 SoftwareSerial 收 57600 是可以跑的，但
 *   中断开销大、遇到大量 raw 时可能丢字节 —— 所以**丢包计数**会随状态帧一起
 *   上报，面板上「丢包 N」一直涨就说明这根线太勉强。要稳就换：
 *     - Arduino Mega / Leonardo（有 Serial1，把 USE_SERIAL1 改成 1）
 *     - ESP32 / RP2040（同上，或直接用 UART1）
 *   接口完全一样，只改一个宏。
 *
 * ---------------------------------------------------------------------------
 * 两个方向的线上格式
 * ---------------------------------------------------------------------------
 *   Arduino → 电脑（同一条 USB 串口，混着两种东西）：
 *     a) TGAM 原始字节流，**原样**，一个字节都不改（电脑侧用 main.py 的
 *        TGAMParser 解析，因此 B 站插件、NeuroFocus 原程序都能直接吃）；
 *     b) 本固件自己的状态帧，固定 16 字节：
 *          AA AA 0C <12 字节 payload> <checksum>
 *          checksum = (~sum(payload)) & 0xFF
 *        payload:
 *          [0] tag     0x01=每秒状态, 0x02=按键 MARK
 *          [1] flags   bit0 link  / bit1 artifact / bit2 btn / bit3 led
 *          [2] attention 0..100
 *          [3] signal    0..200（0 最好）
 *          [4] raw_std / 8      （0..255 → 0..2040）
 *          [5] raw_max / 24     （0..255 → 0..6120）
 *          [6] meditation 0..100
 *          [7] 累计按键次数（截断到 255）
 *          [8] 坏包计数低 8 位
 *          [9] 上一秒 raw 采样数（512 = 正常）
 *          [10] 上一条收到的命令 id
 *          [11] 累计大包数低 8 位
 *        （TGAMParser 看到 plen=0x0C 会当作未知长度帧丢掉，不会污染它的统计）
 *
 *   电脑 → Arduino：纯文本行，以 \n 结尾，一律以 # 开头：
 *          #PING            回一个立即状态帧
 *          #LED:1 / #LED:0  开关片上 LED 反馈
 *          #THR:40          设定 LED 反馈用的 attention 阈值（0..100）
 *          #BLINK           快闪三下（确认链路用）
 *   其他字符一律忽略，所以即使有人在串口监视器里乱打字也不会影响转发。
 * ============================================================================= */

#include <SoftwareSerial.h>

/* ------------------------------- 配置 ----------------------------------- */
#define USE_SERIAL1     0        // 1 = Mega/Leonardo/ESP32/RP2040 用硬件 Serial1
#define TGAM_BAUD       57600
#define LINK_BAUD       115200   // Uno/Leonardo 的 USB CDC 忽略此值，写清楚方便 ESP32

#define PIN_TGAM_RX     2
#define PIN_TGAM_TX     3        // 只占位，永不在用
#define PIN_BTN         4
#define PIN_LED         9
#define PIN_HEART       13

#define LINK_TIMEOUT_MS 2500     // 这么久收不到 TGAM 帧就判定断链
#define LED_THR_DEFAULT 40       // 默认 attention 阈值（0..100）
#define EMIT_PERIOD_MS  1000

#define TAG_STATE       0x01
#define TAG_MARK        0x02

#define FLAG_LINK       0x01
#define FLAG_ARTIFACT   0x02
#define FLAG_BTN        0x04
#define FLAG_LED        0x08

#define ARTIFACT_STD    600      // raw 标准差超过它就标 artifact（眨眼/咬牙/挪动）

/* ------------------------------- 全局状态 -------------------------------- */
#if USE_SERIAL1
  #define TGAM Serial1
#else
  SoftwareSerial TGAM(PIN_TGAM_RX, PIN_TGAM_TX);
#endif

static uint8_t  buf[96];
static uint8_t  blen = 0;

static uint8_t  attention = 0;
static uint8_t  meditation = 0;
static uint8_t  signalQ = 200;
static uint32_t bigCount = 0;
static uint32_t badCount = 0;
static uint32_t lastFrameMs = 0;

static uint16_t rawN = 0;
static float    rawSum = 0.0f;
static float    rawSumSq = 0.0f;
static uint16_t rawPeak = 0;
static float    rawStd = 0.0f;

static uint8_t  marks = 0;
static bool     ledOn = true;
static uint8_t  attThr = LED_THR_DEFAULT;
static uint8_t  lastCmd = 0;
static uint32_t lastEmitMs = 0;
static uint32_t blinkUntil = 0;

static bool     secondDue = false;   // 一秒到了，等一个「帧间空隙」再发状态帧
static uint8_t  pendingTag = 0;      // 排队中的状态帧（0 = 没有）

static bool     btnPrev = false;
static uint32_t btnLastMs = 0;

static char     cmdBuf[24];
static uint8_t  cmdLen = 0;

/* ------------------------------ 小工具 ---------------------------------- */
static void shiftBuf(uint8_t n) {
  if (n >= blen) { blen = 0; return; }
  for (uint8_t i = n; i < blen; i++) buf[i - n] = buf[i];
  blen -= n;
}

/* 收一个字节：既要转发，也要自己看一眼（统计 + 按键判断 + 断链看门狗） */
static void handleFrame(const uint8_t *p, uint8_t plen) {
  lastFrameMs = millis();
  if (plen == 0x20) {                    // 每秒大包
    bigCount++;
    signalQ   = p[1];
    attention = p[29];
    meditation = p[31];
  } else if (plen == 4 && p[0] == 0x80 && p[1] == 0x02) {  // 512 Hz raw
    int16_t v = (int16_t)(((uint16_t)p[2] << 8) | p[3]);
    int16_t a = (v < 0) ? (int16_t)(-v) : v;
    rawN++;
    rawSum   += (float)v;
    rawSumSq += (float)v * (float)v;
    if ((uint16_t)a > rawPeak) rawPeak = (uint16_t)a;
  }
}

static void parseByte(uint8_t b) {
  if (blen >= sizeof(buf)) shiftBuf(1);
  buf[blen++] = b;

  for (;;) {
    if (blen < 3) return;
    if (buf[0] != 0xAA) { shiftBuf(1); continue; }
    if (buf[1] != 0xAA) { shiftBuf(1); continue; }
    uint8_t plen = buf[2];
    if (plen == 0 || plen > 60) { shiftBuf(1); continue; }  // 明显不是帧头
    if (blen < (uint8_t)(plen + 4)) return;                 // 还没收全

    uint8_t sum = 0;
    for (uint8_t i = 0; i < plen; i++) sum += buf[3 + i];
    if (((uint8_t)(~sum)) != buf[3 + plen]) {
      badCount++;
      shiftBuf(1);
      continue;
    }
    handleFrame(&buf[3], plen);
    shiftBuf(plen + 4);
  }
}

/* --------------------------- 状态帧（给电脑） ---------------------------- */
static void emitExt(uint8_t tag) {
  uint8_t pl[12];
  bool linkOk = (millis() - lastFrameMs) < LINK_TIMEOUT_MS;
  bool artifact = (rawStd > (float)ARTIFACT_STD);

  uint8_t flags = 0;
  if (linkOk)   flags |= FLAG_LINK;
  if (artifact) flags |= FLAG_ARTIFACT;
  if (tag == TAG_MARK) flags |= FLAG_BTN;
  if (ledOn)    flags |= FLAG_LED;

  float stdScale = rawStd / 8.0f;
  if (stdScale > 255.0f) stdScale = 255.0f;
  float peakScale = (float)rawPeak / 24.0f;
  if (peakScale > 255.0f) peakScale = 255.0f;

  pl[0]  = tag;
  pl[1]  = flags;
  pl[2]  = attention;
  pl[3]  = signalQ;
  pl[4]  = (uint8_t)stdScale;
  pl[5]  = (uint8_t)peakScale;
  pl[6]  = meditation;
  pl[7]  = marks;
  pl[8]  = (uint8_t)(badCount & 0xFF);
  pl[9]  = (uint8_t)(rawN > 255 ? 255 : rawN);
  pl[10] = lastCmd;
  pl[11] = (uint8_t)(bigCount & 0xFF);

  uint8_t sum = 0;
  Serial.write((uint8_t)0xAA);
  Serial.write((uint8_t)0xAA);
  Serial.write((uint8_t)12);
  for (uint8_t i = 0; i < 12; i++) { Serial.write(pl[i]); sum += pl[i]; }
  Serial.write((uint8_t)(~sum));
}

/* 每秒把 raw 统计结一次账，然后清零重来 */
static void closeSecond() {
  if (rawN > 0) {
    float mean = rawSum / (float)rawN;
    float var  = rawSumSq / (float)rawN - mean * mean;
    rawStd = (var > 0.0f) ? sqrtf(var) : 0.0f;
  } else {
    rawStd = 0.0f;
  }
  rawN = 0; rawSum = 0.0f; rawSumSq = 0.0f; rawPeak = 0;
}

/* 状态帧只能插在「两个 TGAM 帧之间」发：要是插到某个帧的字节中间，
 * 那个 TGAM 帧的校验就废了，电脑侧会看到坏包。所以这里只排队，
 * 主循环发现 blen==0（当前不在帧中途）才真正写出去。 */
static void requestEmit(uint8_t tag) {
  if (pendingTag == 0 || tag == TAG_MARK) pendingTag = tag;
}

static void flushPending() {
  if (pendingTag == 0 || blen != 0) return;
  if (secondDue) {
    closeSecond();
    secondDue = false;
  }
  emitExt(pendingTag);
  pendingTag = 0;
}

/* ------------------------------ 按键 ------------------------------------ */
static void pollButton() {
  bool pressed = (digitalRead(PIN_BTN) == LOW);   // 上拉 + 按下接地
  uint32_t now = millis();
  if (pressed && !btnPrev && (now - btnLastMs) > 250) {
    btnLastMs = now;
    marks++;
    lastCmd = 2;
    requestEmit(TAG_MARK);    // 立刻排队，下一个帧间空隙就发出去（通常 <2 ms）
  }
  btnPrev = pressed;
}

/* ------------------------------ 电脑命令 -------------------------------- */
static void runCmd() {
  if (cmdLen == 0) return;
  cmdBuf[cmdLen] = 0;
  if (cmdBuf[0] != '#') return;           // 只认 # 开头的行

  if (strcmp(cmdBuf, "#PING") == 0) {
    lastCmd = 1;
    requestEmit(TAG_STATE);
  } else if (strcmp(cmdBuf, "#LED:1") == 0) {
    ledOn = true;  lastCmd = 3;
  } else if (strcmp(cmdBuf, "#LED:0") == 0) {
    ledOn = false; lastCmd = 4;
    analogWrite(PIN_LED, 0);
  } else if (strncmp(cmdBuf, "#THR:", 5) == 0) {
    int v = atoi(cmdBuf + 5);
    if (v < 0) v = 0;
    if (v > 100) v = 100;
    attThr = (uint8_t)v;
    lastCmd = 5;
  } else if (strcmp(cmdBuf, "#BLINK") == 0) {
    blinkUntil = millis() + 1200;
    lastCmd = 6;
  }
}

static void pollCommands() {
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c < 0) break;
    if (c == '\n' || c == '\r') {
      runCmd();
      cmdLen = 0;
    } else if (cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = (char)c;
    }
  }
}

/* ------------------------------- LED ------------------------------------ */
static void updateLed() {
  uint32_t now = millis();
  if (now < blinkUntil) {                 // #BLINK：快闪
    analogWrite(PIN_LED, ((now / 120) % 2) ? 255 : 0);
    return;
  }
  if (!ledOn) { analogWrite(PIN_LED, 0); return; }

  // 断链：慢闪（提示头带掉了）；低注意力：呼吸；高注意力：随分数变亮
  if (now - lastFrameMs > LINK_TIMEOUT_MS) {
    analogWrite(PIN_LED, ((now / 400) % 2) ? 70 : 0);
  } else if (attention < attThr) {
    uint16_t phase = (uint16_t)((now / 8) % 256);
    uint16_t tri = (phase < 128) ? phase : (255 - phase);
    analogWrite(PIN_LED, (uint8_t)(30 + (tri * 150) / 128));
  } else {
    int span = 100 - (int)attThr;
    if (span < 1) span = 1;
    int level = 60 + ((int)attention - (int)attThr) * 195 / span;
    if (level > 255) level = 255;
    analogWrite(PIN_LED, (uint8_t)level);
  }
}

/* ------------------------------- setup/loop ----------------------------- */
void setup() {
  Serial.begin(LINK_BAUD);
#if USE_SERIAL1
  Serial1.begin(TGAM_BAUD);
#else
  TGAM.begin(TGAM_BAUD);
  TGAM.listen();
#endif
  pinMode(PIN_BTN, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_HEART, OUTPUT);
  analogWrite(PIN_LED, 0);
  lastFrameMs = millis();
  lastEmitMs  = millis();
  // 开机先发一帧状态，电脑侧一接上就知道对面是 Arduino 桥
  emitExt(TAG_STATE);
}

void loop() {
  pollCommands();

  while (TGAM.available() > 0) {
    uint8_t b = (uint8_t)TGAM.read();
    Serial.write(b);              // 透明转发：一个字节都不改
    parseByte(b);                 // 顺手自己看一遍用于统计/看门狗
  }

  pollButton();

  uint32_t now = millis();
  if ((uint32_t)(now - lastEmitMs) >= EMIT_PERIOD_MS) {
    lastEmitMs = now;
    secondDue = true;
    requestEmit(TAG_STATE);
  }
  flushPending();      // 只在「不在某个 TGAM 帧中途」时才真正写出

  updateLed();

  // 板载心跳：收到帧就快闪，掉线就不亮
  digitalWrite(PIN_HEART, ((now - lastFrameMs) < LINK_TIMEOUT_MS) && ((now / 500) % 2));
}
