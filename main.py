#import the ASGI server uvicorn
import uvicorn

#Entry point for the ASGI server
if __name__ == "__main__":
    #run the server at localhost(0.0.0.0):8000 and enable auto-reload
    #when code changes
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

print("hello world!")
print("how are you?") 