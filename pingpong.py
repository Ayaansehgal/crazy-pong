import cv2
import mediapipe as mp
import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Hand Tracking Ping Pong')
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Font
font = pygame.font.Font(None, 36)

def get_game_limit():
    screen.fill(BLACK)
    text = font.render("Select points to play for: 5, 7, 10", True, WHITE)
    screen.blit(text, (WIDTH // 4, HEIGHT // 3))
    pygame.display.flip()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_5:
                    return 5
                elif event.key == pygame.K_7:
                    return 7
                elif event.key == pygame.K_0:  # 0 represents 10 (since no direct K_10)
                    return 10

game_limit = get_game_limit()

ball = pygame.Rect(WIDTH // 2, HEIGHT // 2, 20, 20)
ball_speed_x, ball_speed_y = 4, 4
paddle_width, paddle_height = 20, 100
left_paddle = pygame.Rect(50, HEIGHT // 2 - paddle_height // 2, paddle_width, paddle_height)
right_paddle = pygame.Rect(WIDTH - 70, HEIGHT // 2 - paddle_height // 2, paddle_width, paddle_height)
left_score, right_score = 0, 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
clock = pygame.time.Clock()

def reset_ball():
    global ball_speed_x, ball_speed_y
    ball.center = (WIDTH // 2, HEIGHT // 2)
    ball_speed_x = 4 if ball_speed_x > 0 else -4
    ball_speed_y = 4 if ball_speed_y > 0 else -4

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    # Camera
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Hand tracking
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * HEIGHT)

            if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                left_paddle.y = max(0, min(HEIGHT - paddle_height, y - paddle_height // 2))
            else:
                right_paddle.y = max(0, min(HEIGHT - paddle_height, y - paddle_height // 2))

    # Ball movement
    ball.x += ball_speed_x
    ball.y += ball_speed_y
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1
    if ball.colliderect(left_paddle) or ball.colliderect(right_paddle):
        ball_speed_x *= -1.1 

    # Scoring
    if ball.left <= 0:
        right_score += 1
        reset_ball()
    elif ball.right >= WIDTH:
        left_score += 1
        reset_ball()
    
    # Check for game over
    if left_score == game_limit or right_score == game_limit:
        screen.fill(BLACK)
        winner_text = font.render(f"Player {'Left' if left_score == game_limit else 'Right'} Wins!", True, WHITE)
        screen.blit(winner_text, (WIDTH // 3, HEIGHT // 3))
        pygame.display.flip()
        pygame.time.delay(3000)
        break

    # Drawing
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, left_paddle)
    pygame.draw.rect(screen, WHITE, right_paddle)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))
    
    # Scores
    left_text = font.render(str(left_score), True, WHITE)
    right_text = font.render(str(right_score), True, WHITE)
    screen.blit(left_text, (WIDTH // 4, 20))
    screen.blit(right_text, (WIDTH * 3 // 4, 20))
    
    pygame.display.flip()
    clock.tick(60)
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
