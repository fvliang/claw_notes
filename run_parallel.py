#!/usr/bin/env python3
import os
os.environ["BAILIAN_API_KEY"] = "sk-sp-H.YHIYD.wmzM.MEYCIQCmxqFd4zS14qwWlEl-7BCckAZ2O3A2M3Vss6KxXw254wIhAIwzFK2OB6MplK1rLa4X06oeAqpn7DQKDDxJnqvKpkyj"
import subprocess
subprocess.run(["python3", "batch_process.py", "--batch-size", "200"])
