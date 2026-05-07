{
  description = "LBSM Research Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default =
        pkgs.mkShell {

          buildInputs = with pkgs; [
            python312
            python312Packages.pip
            python312Packages.numpy
            python312Packages.pandas
            python312Packages.scipy
            python312Packages.scikit-learn
            python312Packages.matplotlib
            python312Packages.seaborn
            python312Packages.jupyterlab
            python312Packages.plotly
            python312Packages.notebook
            python312Packages.ipykernel
          ];

          shellHook = ''
            echo "LBSM research environment ready."
          '';
        };
    };
}