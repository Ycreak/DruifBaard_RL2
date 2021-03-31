class Cart:
    # These are the limits we use for the observation values
    LIM_CART_POSITION = 2.4 # real: 2.4
    LIM_CART_VELOCITY = 1.0 # real: inf
    LIM_POLE_ANGLE = 12     # real: 48
    LIM_POLE_ROTATION = 25  # real: inf

    # These are the step sizes we use
    STEP_CART_POSITION = 0.01
    STEP_CART_VELOCITY = 0.01
    STEP_POLE_ANGLE = 1
    STEP_POLE_ROTATION = 1

    NUMSTEPS_CART_POSITION = int((LIM_CART_POSITION * 2) / STEP_CART_POSITION)
    NUMSTEPS_CART_VELOCITY = int((LIM_CART_VELOCITY * 2) / STEP_CART_VELOCITY)
    NUMSTEPS_POLE_ANGLE = int((LIM_POLE_ANGLE * 2) / STEP_POLE_ANGLE)
    NUMSTEPS_POLE_ROTATION = int((LIM_POLE_ROTATION * 2) / STEP_POLE_ROTATION)

    STATE_SIZE = NUMSTEPS_CART_POSITION * NUMSTEPS_CART_VELOCITY * NUMSTEPS_POLE_ANGLE * NUMSTEPS_POLE_ROTATION
    ACTION_SIZE = 2 # Two actions, 0=push_left, 1=push_right

    def discretize(self, observation) -> int: # Maybe reject values outside selected space, maybe increase precision
        """Map an observation to a unique index, to use for the Q-table

        Args:
            observation (ndarray[4]): [position of cart, velocity of cart, angle of pole, rotation rate of pole]

        Returns:
            int: index based on ordering of the elements in their respective spaces
        """    
        cart_position, cart_velocity, pole_angle, pole_rotation = observation
        
        # calculate orders first
        cart_position = int((round(cart_position, 2) + self.LIM_CART_POSITION) / self.STEP_CART_POSITION)
        cart_velocity = int((round(cart_velocity, 2) + self.LIM_CART_VELOCITY) / self.STEP_CART_VELOCITY)
        pole_angle = int((round(pole_angle, 0) + self.LIM_POLE_ANGLE) / self.STEP_POLE_ANGLE)
        pole_rotation = int((round(pole_rotation, 0) + self.LIM_POLE_ROTATION) / self.STEP_POLE_ROTATION)

        index = cart_position
        index = index + self.NUMSTEPS_CART_POSITION * cart_velocity
        index = index + self.NUMSTEPS_CART_POSITION * self.NUMSTEPS_CART_VELOCITY * pole_angle
        index = index + self.NUMSTEPS_CART_POSITION * self.NUMSTEPS_CART_VELOCITY * self.NUMSTEPS_POLE_ANGLE * pole_rotation

        return index

