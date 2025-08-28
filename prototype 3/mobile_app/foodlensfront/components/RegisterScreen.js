import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Image, Alert } from 'react-native';
import Icon from 'react-native-vector-icons/FontAwesome'; // Import FontAwesome icons
import { useNavigation } from '@react-navigation/native';

const RegisterScreen = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigation = useNavigation();

  // Navigation handlers
  const handleRegister = () => {
    // Basic validation
    if (!name || !email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }
    
    // You can add more validation here (email format, password strength, etc.)
    if (password.length < 6) {
      Alert.alert('Error', 'Password must be at least 6 characters long');
      return;
    }
    
    // Here you would typically call your registration API
    // For now, we'll just navigate to the FoodLens screen
    Alert.alert('Success', 'Registration successful!', [
      {
        text: 'OK',
        onPress: () => navigation.navigate('FoodLens')
      }
    ]);
  };

  const handleSignIn = () => {
    navigation.navigate('Login'); // Navigate to Login screen
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
        {/* Name Input */}
        <View style={styles.inputContainer}>
          <View style={styles.inputWithIcon}>
            <Icon name="user" size={20} color="#d32f2f" style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Name"
              value={name}
              onChangeText={setName}
              autoCapitalize="words"
            />
          </View>
        </View>

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

        {/* Register Button */}
        <TouchableOpacity style={styles.registerButton} onPress={handleRegister}>
          <Text style={styles.registerButtonText}>Register</Text>
        </TouchableOpacity>
           <TouchableOpacity onPress={handleSignIn}>
          <Text style={styles.signInText}>Already have an account? <Text style={styles.signInLink}>Sign In</Text></Text>
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

        {/* Sign In Link */}
     
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
  registerButton: {
    backgroundColor: '#d32f2f',
    width: '100%',
    padding: 15,
    borderRadius: 25,
    alignItems: 'center',
    marginVertical: 10,
  },
  registerButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
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
  signInText: {
    color: '#666',
    fontSize: 14,
    marginTop: 10,
  },
  signInLink: {
    color: '#d32f2f',
    fontWeight: 'bold',
  },

});

export default RegisterScreen;