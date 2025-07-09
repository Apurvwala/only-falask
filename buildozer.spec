[app]

# (str) Title of your application
title = FaceApp Attendance

# (str) Package name
# IMPORTANT: Change 'yourcompany' to something unique to you/your project.
package.name = com.yourcompany.faceappattendance

# (str) Package domain (needed for android/ios packaging)
# IMPORTANT: Change 'yourcompany' to your actual domain.
package.domain = yourcompany.com

# (str) Source code directory. This is the directory that contains your main.py.
source.dir = .

# (list) Application requirements
# This is CRITICAL. List all Python libraries your app uses.
# Kivy is required for the GUI. Flask, numpy, requests, face_recognition, Pillow for your backend logic.
requirements = python3,kivy,flask,numpy,requests,face_recognition,Pillow

# (str) The category of the application
# category = Games # Or 'Business', 'Tools', etc.

# (str) The directory in which to store the build output
build_dir = .buildozer

# (list) Permissions
# These are essential for your app to function on Android.
# INTERNET for Flask/email/Google Forms. CAMERA for face recognition.
# READ/WRITE_EXTERNAL_STORAGE for saving/loading known_faces data.
android.permissions = INTERNET, CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

# (int) Android API levels
# android.api: The Android API level to build against.
# android.minapi: The minimum Android API level your app supports.
# android.targetsdk: The target Android API level.
# Using 33 is generally good for modern apps.
android.api = 33
android.minapi = 21 # Android 5.0 Lollipop
android.targetsdk = 33
# NDK API level (21 = Android 5.0 Lollipop)
android.ndk_api = 21

# (str) Android NDK version (optional, default is latest stable)
# android.ndk = 25b

# (str) Android SDK version (optional, default is latest stable)
# android.sdk = 29

# (list) Android archs to build for
# arm64-v8a is for 64-bit devices (most modern Android phones).
# armeabi-v7a is for 32-bit devices (older phones).
# Building for both is generally recommended for broader compatibility.
android.archs = arm64-v8a, armeabi-v7a

# (bool) If False, the app will not be debuggable
android.debug = True

# (list) Add your app's assets here.
# This ensures 'known_faces' directory and other relevant files are included in the APK.
# source.include_exts: List file extensions to include.
# source.exclude_dirs: List directories to exclude (e.g., build artifacts, git files).
source.include_exts = py,png,jpg,kv,json,mp3
source.exclude_dirs = .buildozer, .git, .github, __pycache__
source.exclude_exts = pyc,pyo

# (str) The name of the main Python file to run
main.py = main.py

# (str) The version code of your application (integer, increments with each release)
version.code = 1

# (str) The version number of your application (e.g., 0.1, 1.0.0)
version = 0.1

# (list) The icon(s) of the application (optional)
# icon.filename = icon.png # If you have an icon.png in your root directory

# (bool) Enable/disable Android logging (useful for debugging on device)
android.logcat = True

# (bool) Enable/disable Android multiprocessing (often needed for complex apps)
android.multiprocess = True

# (bool) Enable/disable Android hardware acceleration
android.hwaccel = True

# (bool) If True, the app will be launched in immersive mode (full screen)
android.immersive_mode = True

# (bool) If True, the app will request all permissions at runtime
# This is good practice for modern Android versions.
android.request_all_permissions = True

# (str) The default orientation of the screen
# orientation = portrait # or 'landscape' or 'all'
