import getpass
from finance.security import SecureVault
import sys

def main():
    try:
        vault = SecureVault()
    except ValueError as e:
        print(f"Error initializing Vault: {e}")
        sys.exit(1)

    print("=== Secure Vault Setup ===")
    user_id = input("Enter User ID (e.g., portfolio_manager_1): ").strip()
    login = input("Enter Broker/Exchange Login: ").strip()
    
    print("\n[Security] Input your keys. Keystrokes will be hidden.")
    api_key    = getpass.getpass("Enter API Key (Public Key): ").strip()
    secret_key = getpass.getpass("Enter Secret Key (Private Key): ").strip()
    
    if not all([user_id, login, api_key, secret_key]):
        print("Error: All fields are required.")
        sys.exit(1)

    vault.save_user_keys(
        user_id=user_id, 
        login=login, 
        api_key=api_key,
        secret_key=secret_key,
    )
    print("Success: Ключи надежно зашифрованы и сохранены в базу данных.")

if __name__ == "__main__":
    main()