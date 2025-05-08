using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class CameraMovement : MonoBehaviour
{
    public float speed = 2.0f;
    public float mouseSensitivity = 1024.0f;

    // Variables to store camera rotation
    private float rotationX = 0f;
    private float rotationY = 0f;

    public bool movementEnabled = false;

    void Start()
    {
        // Initialize rotationX and rotationY based on the camera's current rotation
        Vector3 currentRotation = transform.localRotation.eulerAngles;
        rotationX = currentRotation.x;
        rotationY = currentRotation.y;

        enabled = movementEnabled;
    }

    void Update()
    {
        if (Input.GetMouseButton((int)MouseButton.LeftMouse))
        {
            float dt = Mathf.Min(Time.deltaTime, 0.1f);
            // Mouse input for looking around
            rotationX -= Mathf.Clamp(Input.GetAxis("Mouse Y"), -1.0f, 1.0f) * mouseSensitivity * dt;
            rotationY += Mathf.Clamp(Input.GetAxis("Mouse X"), -1.0f, 1.0f) * mouseSensitivity * dt;

            // Clamp the up/down rotation to prevent the camera from flipping over
            rotationX = Mathf.Clamp(rotationX, -90f, 90f);

            // Apply the rotation to the camera
            transform.localRotation = Quaternion.Euler(rotationX, rotationY, 0f);
        }

        Vector3 moveDirection = Vector3.zero;
        if (Input.GetKey(KeyCode.A)) moveDirection.x -= 1f;
        if (Input.GetKey(KeyCode.D)) moveDirection.x += 1f;
        if (Input.GetKey(KeyCode.S)) moveDirection.z -= 1f;
        if (Input.GetKey(KeyCode.W)) moveDirection.z += 1f;
	
        if (moveDirection.sqrMagnitude > 0.001f)
        {
            moveDirection.Normalize();

            Vector3 vel = speed * Time.deltaTime * moveDirection;

            transform.Translate(vel);
        }
    }
}
