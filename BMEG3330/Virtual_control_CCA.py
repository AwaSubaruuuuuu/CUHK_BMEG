import sys
import time
import math
import threading
import numpy as np
import pygame
from collections import deque
from scipy import signal
from sklearn.cross_decomposition import CCA
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

################### [YOU CAN CHANGE] - You can change hardware and decoding parameters here if possible ###################
SERIAL_PORT = "COM6"    
FS = 200  # DO NOT CHANGE this unless you know what you are doing, it should match the sampling rate of your OpenBCI board              
WIN_TIME = 3.0  # [YOU CAN CHANGE] you can try different window lengths to find the best matching
SLIDE_STEP = 0.2  #  [YOU CAN CHANGE] you can try different sliding steps to find the most comfortable one for you (smaller step means more frequent updates but higher computational load)        
THRESHOLD = 0.135  #  [YOU CAN CHANGE] you can decrease tbe threshold if u find the car is too hard to move, or increase it if you find the car is too sensitive to noise (you can also try different thresholds for different frequencies, but here we use a single threshold for simplicity)      
BUFFER_SIZE = 2         

################### [YOU CAN CHANGE] - You can try different target frequencies and directional mapping to find the best matching ###################
TARGETS = {
    8.0:  "UP",
    11.0: "DOWN",
    13.0: "LEFT",
    15.0: "RIGHT"
}
FREQS = list(TARGETS.keys())
FB_WEIGHTS = [1.0, 0.42, 0.25]

################### [DO NOT CHANGE] - Global state variables ###################
current_command = "STOP"
fbcca_current_score = 0.0
is_running = True

################### [DO NOT CHANGE] - FBCCA Algorithms ###################
def build_reference_signals(win_len, fs, freqs):
    t = np.arange(win_len) / fs
    Y_ref = []
    for f in freqs:
        ref = []
        for h in range(1, 4): 
            ref.append(np.sin(2 * np.pi * h * f * t))
            ref.append(np.cos(2 * np.pi * h * f * t))
        Y_ref.append(np.array(ref).T)
    return Y_ref

def get_filter_banks(fs):
    banks = []
    b1, a1 = signal.butter(4, [6.0/(fs/2), 45.0/(fs/2)], 'bandpass')
    b2, a2 = signal.butter(4, [14.0/(fs/2), 45.0/(fs/2)], 'bandpass')
    b3, a3 = signal.butter(4, [22.0/(fs/2), 45.0/(fs/2)], 'bandpass')
    banks.extend([(b1, a1), (b2, a2), (b3, a3)])
    return banks

def get_fbcca_score(trial_data_raw, Y_ref, filter_banks, weights):
    fbcca_score = 0.0
    for i, (b, a) in enumerate(filter_banks):
        filtered_data = signal.filtfilt(b, a, trial_data_raw, axis=1)
        X = filtered_data.T 
        cca = CCA(n_components=1)
        try:
            cca.fit(X, Y_ref)
            Xc, Yc = cca.transform(X, Y_ref)
            r = np.corrcoef(Xc[:, 0], Yc[:, 0])[0, 1]
        except:
            r = 0
        fbcca_score += weights[i] * (r ** 2)
    return fbcca_score

################### [DO NOT CHANGE] - Background BCI Decoding ###################
def bci_decoding_thread(board):
    global current_command, fbcca_current_score, is_running
    
    eeg_channels = BoardShim.get_eeg_channels(board.board_id)[:4]
    win_len = int(WIN_TIME * FS)
    
    filter_banks = get_filter_banks(FS)
    Y_ref = build_reference_signals(win_len, FS, FREQS)
    decision_buffer = deque(maxlen=BUFFER_SIZE)
    
    while is_running:
        start_time = time.time()
        current_data = board.get_current_board_data(win_len)
        
        if current_data.shape[1] >= win_len:
            eeg_data = current_data[eeg_channels, :]
            eeg_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
            
            scores = []
            for f_idx in range(len(FREQS)):
                score = get_fbcca_score(eeg_data, Y_ref[f_idx], filter_banks, FB_WEIGHTS)
                scores.append(score)
                
            max_score = np.max(scores)
            pred_idx = np.argmax(scores)
            fbcca_current_score = max_score
            
            if max_score >= THRESHOLD:
                raw_cmd = TARGETS[FREQS[pred_idx]]
                decision_buffer.append(raw_cmd)
            else:
                decision_buffer.append("STOP")
            
            if len(decision_buffer) == BUFFER_SIZE and len(set(decision_buffer)) == 1:
                current_command = decision_buffer[0]
            else:
                current_command = "STOP"
                
            print(f"[Decoder] Output: {current_command:<5} | Max Score: {max_score:.3f}")
        
        elapsed = time.time() - start_time
        sleep_time = max(0, SLIDE_STEP - elapsed)
        time.sleep(sleep_time)

################### [YOU CAN CHANGE] - You can change the virtual car's visual properties here ###################
class VirtualCar:
    def __init__(self, x, y, bounds_w, bounds_h):
        self.x = x
        self.y = y
        self.bounds_w = bounds_w
        self.bounds_h = bounds_h
        self.size = 40
        self.speed = 3.0 
        self.color = (0, 255, 100) 

    ################### [DO NOT CHANGE] - Driving and collision logic ###################
    def update(self, command):
        if command == "UP":
            self.y -= self.speed
        elif command == "DOWN":
            self.y += self.speed
        elif command == "LEFT":
            self.x -= self.speed
        elif command == "RIGHT":
            self.x += self.speed
            
        self.x = max(200, min(self.bounds_w - 200, self.x))
        self.y = max(200, min(self.bounds_h - 200, self.y))

    def draw(self, surface):
        rect = pygame.Rect(self.x - self.size//2, self.y - self.size//2, self.size, self.size)
        pygame.draw.rect(surface, self.color, rect)
        pygame.draw.rect(surface, (255, 255, 255), rect, 3) 

################### [DO NOT CHANGE] - Main UI ###################
def main():
    global is_running
    
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BoardIds.GANGLION_NATIVE_BOARD.value, params)
    
    print("Connecting to OpenBCI Ganglion...")
    board.prepare_session()
    board.start_stream()
    time.sleep(3) 
    
    decoder_thread = threading.Thread(target=bci_decoding_thread, args=(board,))
    decoder_thread.daemon = True
    decoder_thread.start()
    
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    pygame.display.set_caption("SSVEP Virtual Car Controller")
    font = pygame.font.Font(None, 45) 
    small_font = pygame.font.Font(None, 30)
    
    screen_w, screen_h = screen.get_width(), screen.get_height()
    cx, cy = screen_w // 2, screen_h // 2
    
    offset_x, offset_y = 450, 300      
    box_size = 150    
    half_box = box_size // 2

    rects = {
        8.0:  pygame.Rect(cx - half_box, cy - offset_y - half_box, box_size, box_size), 
        11.0: pygame.Rect(cx - half_box, cy + offset_y - half_box, box_size, box_size), 
        13.0: pygame.Rect(cx - offset_x - half_box, cy - half_box, box_size, box_size), 
        15.0: pygame.Rect(cx + offset_x - half_box, cy - half_box, box_size, box_size)  
    }

    car = VirtualCar(cx, cy, screen_w, screen_h)
    clock = pygame.time.Clock()
    start_time = time.time()
    
    try:
        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    is_running = False
            
            screen.fill((20, 20, 20)) 
            
            pygame.draw.rect(screen, (50, 50, 50), (200, 200, screen_w-400, screen_h-400), 2)
            
            car.update(current_command)
            car.draw(screen)
            
            elapsed = time.time() - start_time
            for f, rect in rects.items():
                luminance = 0.5 * (1 + math.sin(2 * math.pi * f * elapsed))
                color_val = int(255 * luminance)
                pygame.draw.rect(screen, (color_val, color_val, color_val), rect)
                
                label_text = font.render(TARGETS[f], True, (200, 200, 200))
                screen.blit(label_text, (rect.x + box_size//2 - label_text.get_width()//2, rect.y - 40))
                
            info_str = f"COMMAND: {current_command} | SCORE: {fbcca_current_score:.3f} | WIN: {WIN_TIME}s"
            info_color = (0, 255, 0) if current_command != "STOP" else (150, 150, 150)
            info_surface = font.render(info_str, True, info_color)
            screen.blit(info_surface, (cx - info_surface.get_width()//2, 50))
            
            pygame.display.flip()
            clock.tick(60) 

    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        print("Shutting down... Releasing OpenBCI resources.")
        is_running = False
        board.stop_stream()
        board.release_session()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()