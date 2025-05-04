// index.js

const http = require("http");

// Create a simple server
const server = http.createServer((req, res) => {
  res.statusCode = 200; // OK
  res.setHeader("Content-Type", "text/plain");
  res.end("Hello, World!\n");
});

// Set the port
const PORT = 4000;

// Start the server
server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});// index.js
