interface DisclaimerBannerProps {
  text: string;
}

export function DisclaimerBanner({ text }: DisclaimerBannerProps) {
  return (
    <div className="mx-auto max-w-3xl px-6 pb-6 pt-3 text-center">
      <p className="text-[11px] leading-5 text-gray-400">
        {text}
      </p>
    </div>
  );
}
