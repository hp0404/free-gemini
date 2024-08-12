# free-gemini

This repository contains a Python script for interacting with the Google Gemini Flash model with built-in rate limiting. The script allows you to generate text content while respecting predefined API usage limits, helping to prevent overuse and potential service disruptions.

Note that as implemented, the example expects the model to return responses in JSON format, which are then parsed into a structured dictionary and can be further transformed into a tabular format.
