import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Image, Alert } from 'react-native';
import Icon from 'react-native-vector-icons/FontAwesome'; // Import FontAwesome icons
import { useNavigation } from '@react-navigation/native';

const LoginScreen = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigation = useNavigation();

  // Navigation handlers
  // const handleLogin = () => {
  //   // Basic validation
  //   if (!email || !password) {
  //     Alert.alert('Error', 'Please fill in all fields');
  //     return;
  //   }
    
  //   // Basic email validation
  //   const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  //   if (!emailRegex.test(email)) {
  //     Alert.alert('Error', 'Please enter a valid email address');
  //     return;
  //   }
    
  //   // Here you would typically call your login API
  //   // For now, we'll just navigate to the FoodLens screen
  //   Alert.alert('Success', 'Login successful!', [
  //     {
  //       text: 'OK',
  //       onPress: () => navigation.navigate('FoodLens')
  //     }
  //   ]);
  // };


  const handleLogin = () => {
    navigation.navigate('FoodLens');
  }

  const handleSignUp = () => {
    navigation.navigate('Register'); // Navigate to Register screen
  };

  const handleForgotPassword = () => {
    // You can create a forgot password screen or show an alert
    Alert.alert('Forgot Password', 'Forgot password functionality would be implemented here');
  };

  const handleGoogleLogin = () => {
    // Implement Google login logic here
    Alert.alert('Google Login', 'Google login functionality would be implemented here');
  };

  const handleFacebookLogin = () => {
    // Implement Facebook login logic here
    Alert.alert('Facebook Login', 'Facebook login functionality would be implemented here');
  };

  return (
    <View style={styles.container}>
      {/* Header Section with Curved Bottom */}
      <View style={styles.header}>
        {/* Logo */}
        <View style={styles.logoContainer}>
          <Image
            source={require('../assets/whiteeLogo.png')} // Replace with actual logo image path
            style={styles.logo}
            resizeMode="contain"
          />
        </View>
     
        {/* Curved Bottom Effect */}
        <View style={styles.curveContainer}>
          <View style={styles.curve} />
        </View>
        
        <View style={styles.containIcon}>
          <Image
            source={require('../assets/foodicon.png')} // Replace with actual logo image path
            style={styles.iconCollection}
            resizeMode="contain"
          />
        </View>
      </View>

      {/* Form Section */}
      <View style={styles.formContainer}>
        {/* Email Input */}
        <View style={styles.inputContainer}>
          <View style={styles.inputWithIcon}>
            <Icon name="envelope" size={20} color="#d32f2f" style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Email"
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              autoCapitalize="none"
            />
          </View>
        </View>

        {/* Password Input */}
        <View style={styles.inputContainer}>
          <View style={styles.inputWithIcon}>
            <Icon name="lock" size={20} color="#d32f2f" style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Password"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />
          </View>
        </View>

        {/* Login Button */}
        <TouchableOpacity style={styles.loginButton} onPress={handleLogin}>
          <Text style={styles.loginButtonText}>Login</Text>
        </TouchableOpacity>

        {/* Forgot Password */}
        <TouchableOpacity onPress={handleForgotPassword}>
          <Text style={styles.forgotPassword}>Forgot your Password?</Text>
        </TouchableOpacity>

        {/* Divider */}
        <View style={styles.divider}>
          <View style={styles.line} />
          <Text style={styles.orText}>or</Text>
          <View style={styles.line} />
        </View>

        {/* Social Login Buttons */}
        <View style={styles.socialButtons}>
          <TouchableOpacity style={styles.socialButton} onPress={handleGoogleLogin}>
            <Icon name="google" size={20} color="#d32f2f" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.socialButton} onPress={handleFacebookLogin}>
            <Icon name="facebook" size={20} color="#d32f2f" />
          </TouchableOpacity>
        </View>

        {/* Sign Up Link */}
        <TouchableOpacity onPress={handleSignUp}>
          <Text style={styles.signUpText}>Don't have an account? <Text style={styles.signUpLink}>Sign Up</Text></Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  header: {
    backgroundColor: '#d32f2f',
    height: 400,
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
    borderBottomLeftRadius: 40,
    borderBottomRightRadius: 40,
  },
  logoContainer: {
    width: 100,
    height: 100,
    borderColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 300,
  },
  logo: {
    width: '100%',
    height: '100%',
  },
  containIcon: {
    marginBottom: 70,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 78,
  },
  iconCollection: {
    width: 430,
    height: 300,
  },
  curveContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 50,
    justifyContent: 'flex-end',
  },
  curve: {
    height: 50,
    backgroundColor: '#d32f2f',
    borderBottomLeftRadius: 100,
    borderBottomRightRadius: 100,
    transform: [{ scaleX: 1.1 }],
  },
  formContainer: {
    flex: 1,
    padding: 20,
    alignItems: 'center',
  },
  inputContainer: {
    width: '100%',
    marginBottom: 15,
  },
  inputWithIcon: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 25,
    borderWidth: 1,
    borderColor: '#ddd',
    paddingHorizontal: 15,
  },
  inputIcon: {
    marginRight: 10,
  },
  input: {
    flex: 1,
    padding: 15,
    fontSize: 16,
  },
  loginButton: {
    backgroundColor: '#d32f2f',
    width: '100%',
    padding: 15,
    borderRadius: 25,
    alignItems: 'center',
    marginVertical: 10,
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  forgotPassword: {
    color: '#d32f2f',
    fontSize: 14,
    marginVertical: 10,
    textDecorationLine: 'underline',
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 20,
  },
  line: {
    flex: 1,
    height: 1,
    backgroundColor: '#ccc',
  },
  orText: {
    marginHorizontal: 10,
    color: '#666',
  },
  socialButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '50%',
    marginVertical: 10,
  },
  socialButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  signUpText: {
    color: '#666',
    fontSize: 14,
    marginTop: 10,
  },
  signUpLink: {
    color: '#d32f2f',
    fontWeight: 'bold',
  },
});

export default LoginScreen;