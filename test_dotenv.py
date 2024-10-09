import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv('ITERS'))
print(os.getenv('GUMBEL'))
print(os.getenv('CHANNELS'))

OUTPUT_DIM = 784 * int(os.getenv('CHANNELS'))
print(OUTPUT_DIM)