#!/usr/bin/env python3
import base64
import signal
import time
from typing import Optional
from src.hand_signs import HandSigns

import cv2
import numpy as np
from fastapi import Response
from nicegui import Client, app, core, run, ui

black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')

def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()

def setup() -> None:
    hand_signs = HandSigns()
    video_capture = cv2.VideoCapture(0)
    last_frame: Optional[np.ndarray] = None
    preview_bytes: Optional[bytes] = None
    is_test_started = False
    is_capturing = True 
    hands_detected = False

    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        nonlocal last_frame
        if not video_capture.isOpened() or not is_capturing:
            return placeholder # do not capture if webcam is paused
        ret, frame = await run.io_bound(video_capture.read)
        if not ret or frame is None:
            return placeholder
        last_frame = frame.copy()  # Store the last valid frame for later use

        hand_signs.detect_async(frame)
        if hand_signs.result:
            frame = hand_signs.draw_landmarks_on_image(frame, hand_signs.result)
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')
    
    async def capture_photo():
        nonlocal preview_bytes, is_capturing, hands_detected
        if last_frame is not None and last_frame.size > 1:
            # pause webcam capture
            is_capturing = False
            video_image.classes('hidden')

            # run hand detection
            hand_signs.detect_async(last_frame)
            if hand_signs.result.hand_landmarks:
                hands_detected = True
                # annotate the captured frame with landmarks
                annotated_frame = hand_signs.draw_landmarks_on_image(last_frame, hand_signs.result)
                preview_bytes = await run.cpu_bound(convert, annotated_frame)

                # show the preview image
                preview_image.set_source(f'data:image/jpeg;base64,{base64.b64encode(preview_bytes).decode()}')
                label_input.classes(remove='hidden')
                submit_button.classes(remove='hidden')
            else:
                hands_detected = False 
                ui.notify('No hands detected, please try again.')
                preview_bytes = await run.cpu_bound(convert, last_frame)
                preview_image.set_source(f'data:image/jpeg;base64,{base64.b64encode(preview_bytes).decode()}')
            
            preview_image.classes(remove='hidden')
            retry_button.classes(remove='hidden')

            capture_button.classes('hidden')
            start_stop_button.classes('hidden')
            reset_button.classes('hidden')

        else:
            ui.notify('No frame available.')

    async def submit_photo():
        nonlocal preview_bytes, hands_detected
        if not preview_bytes or not label_input.value.strip():
            ui.notify('Missing photo or label')
            return
        if not hands_detected:
            ui.notify('No hand sign detected. Please retry.')
            return
        
        label = label_input.value.strip().replace(" ", "_")

        hand_signs.store_to_db(label)
        ui.notify(f'Saved {label} sign to db')
        reset_ui()

    async def reset_db():
        hand_signs.reset_db()

    def update_submit_state():
        nonlocal preview_bytes, hands_detected
        if preview_bytes and label_input.value.strip() and hands_detected:
            submit_button.enable()
        else:
            submit_button.disable()

    async def retry_capture():
        nonlocal hands_detected, is_capturing
        hands_detected = False  
        preview_image.set_source('') 
        
        retry_button.classes('hidden')  
        label_input.classes('hidden')  
        submit_button.classes('hidden') 
        reset_button.classes('hidden')
        
        video_image.classes(remove='hidden')
        capture_button.classes(remove='hidden')
        start_stop_button.classes(remove='hidden')
        
        is_capturing = True

    async def start_stop_test():
        nonlocal is_test_started

        if is_test_started:
            # Stop the test: show capture button and change the text
            capture_button.classes(remove='hidden') 
            reset_button.classes(remove='hidden')
            start_stop_button.set_text('Start Test')  
            hand_signs.testing = False
        else:
            # Start the test: hide capture button and change the text
            capture_button.classes('hidden') 
            reset_button.classes('hidden')
            start_stop_button.set_text('Stop Test') 
            hand_signs.testing = True

        is_test_started = not is_test_started

    def reset_ui():
        nonlocal is_capturing, hands_detected
        video_image.classes(remove='hidden')
        capture_button.classes(remove='hidden')
        start_stop_button.classes(remove='hidden')
        reset_button.classes(remove='hidden')
        label_input.set_value('')
        label_input.classes('hidden')
        submit_button.classes('hidden')
        preview_image.set_source('')
        preview_image.classes('hidden')
        submit_button.disable()
        retry_button.classes('hidden')
        is_capturing = True

    with ui.column().classes('justify-center w-full'):
        with ui.element('div').classes('relative mx-auto max-w-xl w-full aspect-video'):
            video_image = ui.interactive_image().classes('absolute inset-0 w-full h-full object-cover')
            ui.timer(interval=0.1, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))
            preview_image = ui.image().classes('absolute inset-0 w-full h-full object-cover hidden')

        with ui.row().classes('justify-center w-full mt-4'):
            start_stop_button = ui.button('Start Test', on_click=start_stop_test).classes('w-fit mt-2')
            capture_button = ui.button('Capture Photo', on_click=capture_photo).classes('w-fit mt-2')
            reset_button = ui.button('Reset DB', on_click=reset_db).props('color=red').classes('w-fit mt-2 bg-red-500 text-white hover:bg-red-600')
            label_input = ui.input(placeholder='Enter label for photo', on_change=lambda _: update_submit_state()).classes('mt-2 hidden w-64')
            submit_button = ui.button('Submit & Save Photo', on_click=submit_photo).classes('hidden mt-2 w-fit')
            submit_button.disable()
            retry_button = ui.button('Retry', on_click=retry_capture).classes('hidden mt-2 w-fit')

    async def disconnect() -> None:
        for client_id in Client.instances:
            await core.sio.disconnect(client_id)

    def handle_sigint(signum, frame) -> None:
        ui.timer(0.1, disconnect, once=True)
        ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)

    async def cleanup() -> None:
        await disconnect()
        video_capture.release()
        hand_signs.close()

    app.on_shutdown(cleanup)
    signal.signal(signal.SIGINT, handle_sigint)

app.on_startup(setup)
ui.run()
