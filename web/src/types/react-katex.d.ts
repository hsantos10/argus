declare module "react-katex" {
  import type { ComponentType, ReactNode } from "react";

  export interface KatexComponentProps {
    math?: string;
    children?: ReactNode;
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
    settings?: Record<string, unknown>;
    as?: string;
  }

  export const InlineMath: ComponentType<KatexComponentProps>;
  export const BlockMath: ComponentType<KatexComponentProps>;
}
